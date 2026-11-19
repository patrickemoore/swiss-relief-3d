//! Single files out of a zip archive on a web server, without downloading
//! the archive. The central directory at the end of the archive says where
//! each file's bytes are; those are fetched with a range request and
//! inflated on the way to disk.
//!
//! swissTLM3D comes as one 3.6 GB zip of which the hiking trails need three
//! files, so this saves most of the download.

use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use miniz_oxide::inflate::stream::{inflate, InflateState};
use miniz_oxide::{DataFormat, MZError, MZFlush, MZStatus};
use ureq::Agent;

/// Bytes read from the end of the archive to find the central directory:
/// the end record, its comment and, in a large archive, the zip64 records.
const TAIL_BYTES: u64 = 2 * 1024 * 1024;
const CHUNK: usize = 1 << 20;

#[derive(Clone, Debug)]
pub struct Entry {
    pub name: String,
    pub compressed: u64,
    pub uncompressed: u64,
    /// 0 for stored, 8 for deflated.
    method: u16,
    header_offset: u64,
}

fn u16_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("eight bytes"))
}

fn fetch_range(agent: &Agent, url: &str, start: u64, end_inclusive: u64) -> Result<impl Read> {
    let response = agent
        .get(url)
        .set("Range", &format!("bytes={start}-{end_inclusive}"))
        .call()
        .with_context(|| format!("fetching {url}"))?;
    ensure!(response.status() == 206, "{url}: the server ignored the range request (HTTP {})", response.status());
    Ok(response.into_reader())
}

fn content_length(agent: &Agent, url: &str) -> Result<u64> {
    let response = agent.head(url).call().with_context(|| format!("asking the size of {url}"))?;
    response.header("Content-Length").and_then(|value| value.parse().ok()).with_context(|| format!("{url}: no size"))
}

/// Lists the files in the archive at `url`.
pub fn list(agent: &Agent, url: &str) -> Result<Vec<Entry>> {
    let length = content_length(agent, url)?;
    let tail_start = length.saturating_sub(TAIL_BYTES);
    let mut tail = Vec::new();
    fetch_range(agent, url, tail_start, length - 1)?.read_to_end(&mut tail)?;

    // The end of central directory record, searched for from the back.
    let eocd = tail
        .windows(4)
        .rposition(|window| window == [0x50, 0x4b, 0x05, 0x06])
        .context("no end-of-central-directory record: not a zip archive")?;
    let mut entries = u16_at(&tail, eocd + 10) as u64;
    let mut directory_size = u32_at(&tail, eocd + 12) as u64;
    let mut directory_offset = u32_at(&tail, eocd + 16) as u64;
    if entries == 0xffff || directory_size == 0xffff_ffff || directory_offset == 0xffff_ffff {
        // Zip64: a locator just before the record points at the zip64 record.
        let locator = eocd.checked_sub(20).filter(|&at| tail[at..at + 4] == [0x50, 0x4b, 0x06, 0x07]);
        let locator = locator.context("a zip64 archive without its locator")?;
        let zip64_offset = u64_at(&tail, locator + 8);
        let at = (zip64_offset.checked_sub(tail_start)).context("the zip64 record lies outside the tail")? as usize;
        ensure!(tail[at..at + 4] == [0x50, 0x4b, 0x06, 0x06], "a malformed zip64 record");
        entries = u64_at(&tail, at + 32);
        directory_size = u64_at(&tail, at + 40);
        directory_offset = u64_at(&tail, at + 48);
    }

    let directory: Vec<u8> = if directory_offset >= tail_start {
        let at = (directory_offset - tail_start) as usize;
        tail[at..at + directory_size as usize].to_vec()
    } else {
        let mut bytes = Vec::with_capacity(directory_size as usize);
        fetch_range(agent, url, directory_offset, directory_offset + directory_size - 1)?.read_to_end(&mut bytes)?;
        bytes
    };

    let mut list = Vec::with_capacity(entries as usize);
    let mut at = 0;
    while at + 46 <= directory.len() && directory[at..at + 4] == [0x50, 0x4b, 0x01, 0x02] {
        let method = u16_at(&directory, at + 10);
        let mut compressed = u32_at(&directory, at + 20) as u64;
        let mut uncompressed = u32_at(&directory, at + 24) as u64;
        let name_length = u16_at(&directory, at + 28) as usize;
        let extra_length = u16_at(&directory, at + 30) as usize;
        let comment_length = u16_at(&directory, at + 32) as usize;
        let mut header_offset = u32_at(&directory, at + 42) as u64;
        let name = String::from_utf8_lossy(&directory[at + 46..at + 46 + name_length]).replace('\\', "/");
        // Sizes and offsets that overflow 32 bits are in a zip64 extra field.
        let extra = &directory[at + 46 + name_length..at + 46 + name_length + extra_length];
        let mut e = 0;
        while e + 4 <= extra.len() {
            let (id, size) = (u16_at(extra, e), u16_at(extra, e + 2) as usize);
            if id == 0x0001 {
                let mut f = e + 4;
                for value in [&mut uncompressed, &mut compressed, &mut header_offset] {
                    if *value == 0xffff_ffff && f + 8 <= e + 4 + size {
                        *value = u64_at(extra, f);
                        f += 8;
                    }
                }
            }
            e += 4 + size;
        }
        list.push(Entry { name, compressed, uncompressed, method, header_offset });
        at += 46 + name_length + extra_length + comment_length;
    }
    Ok(list)
}

/// Inflates a raw deflate stream from `reader` into `out`.
fn inflate_to(reader: &mut impl Read, out: &mut impl Write, name: &str) -> Result<()> {
    let mut state = InflateState::new(DataFormat::Raw);
    let mut input = vec![0u8; CHUNK];
    let mut output = vec![0u8; 4 * CHUNK];
    let mut buffered = 0;
    loop {
        let read = reader.read(&mut input[buffered..])?;
        ensure!(read > 0 || buffered > 0, "{name}: the download ended early");
        buffered += read;
        let mut consumed = 0;
        loop {
            let result = inflate(&mut state, &input[consumed..buffered], &mut output, MZFlush::None);
            consumed += result.bytes_consumed;
            out.write_all(&output[..result.bytes_written])?;
            match result.status {
                Ok(MZStatus::StreamEnd) => return Ok(()),
                // Either the input is used up or the output was filled; only
                // the second case has more to give without more input.
                Ok(_) | Err(MZError::Buf) => {
                    if result.bytes_written < output.len() {
                        break;
                    }
                }
                Err(error) => bail!("{name}: inflating failed: {error:?}"),
            }
        }
        input.copy_within(consumed..buffered, 0);
        buffered -= consumed;
        if read == 0 && consumed == 0 {
            bail!("{name}: the download ended early");
        }
    }
}

/// Fetches one file of the archive to `dest`, replacing what is there.
pub fn fetch(agent: &Agent, url: &str, entry: &Entry, dest: &Path) -> Result<()> {
    // The local header repeats the name and may carry its own extra field;
    // the data follows both.
    let mut header = [0u8; 30];
    fetch_range(agent, url, entry.header_offset, entry.header_offset + 29)?.read_exact(&mut header)?;
    ensure!(header[..4] == [0x50, 0x4b, 0x03, 0x04], "{}: a malformed local header", entry.name);
    let data_start = entry.header_offset + 30 + u16_at(&header, 26) as u64 + u16_at(&header, 28) as u64;
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    let part = dest.with_extension("part");
    let mut out = BufWriter::new(File::create(&part).with_context(|| format!("creating {}", part.display()))?);
    let mut reader = fetch_range(agent, url, data_start, data_start + entry.compressed - 1)?;
    match entry.method {
        0 => {
            std::io::copy(&mut reader, &mut out)?;
        }
        8 => inflate_to(&mut reader, &mut out, &entry.name)?,
        other => bail!("{}: compression method {other} is not supported", entry.name),
    }
    out.flush()?;
    drop(out);
    fs::rename(&part, dest).with_context(|| format!("moving {} into place", part.display()))?;
    Ok(())
}

/// Makes sure the files `names` (matched as suffixes of the archive's paths)
/// are in `dir`, fetching those that are not. Returns their paths in order.
pub fn ensure_files(agent: &Agent, url: &str, dir: &Path, names: &[&str]) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::with_capacity(names.len());
    let mut listed: Option<Vec<Entry>> = None;
    for name in names {
        let file = Path::new(name).file_name().context("a file name")?;
        let path = dir.join(file);
        if !path.exists() {
            let entries = match &listed {
                Some(entries) => entries,
                None => listed.insert(list(agent, url)?),
            };
            let entry = entries
                .iter()
                .find(|entry| entry.name.ends_with(name))
                .with_context(|| format!("{name} is not in the archive"))?;
            println!(
                "fetching {} ({:.0} MB compressed, {:.0} MB unpacked)",
                entry.name,
                entry.compressed as f64 / 1e6,
                entry.uncompressed as f64 / 1e6
            );
            fetch(agent, url, entry, &path)?;
        }
        paths.push(path);
    }
    Ok(paths)
}
