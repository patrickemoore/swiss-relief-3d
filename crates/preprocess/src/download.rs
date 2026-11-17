//! Parallel, resumable tile download.

use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, ensure, Context, Result};

use crate::geotiff;
use crate::stac::TileRef;

const ATTEMPTS: u32 = 4;
/// Bytes requested first when only a tile's overview is wanted. The overview
/// sits right behind the header and ends within this prefix for almost every
/// tile; the rest is fetched with a second request when it does not.
const OVERVIEW_PREFIX_BYTES: usize = 270_336;

/// An HTTP agent on the platform's TLS stack.
pub fn http_agent() -> Result<ureq::Agent> {
    let tls = native_tls::TlsConnector::new().context("initializing TLS")?;
    Ok(ureq::AgentBuilder::new()
        .tls_connector(Arc::new(tls))
        .timeout_connect(Duration::from_secs(20))
        .timeout(Duration::from_secs(180))
        .build())
}

/// Downloads every tile in full into `dir` and returns the local paths.
///
/// Tiles already on disk are kept, so an interrupted run resumes where it
/// stopped.
pub fn download_tiles(agent: &ureq::Agent, tiles: &[TileRef], dir: &Path, jobs: usize) -> Result<Vec<PathBuf>> {
    download_all(tiles, dir, "2m", jobs, |url, path| {
        let mut reader = agent.get(url).call()?.into_reader();
        write_atomically(path, |writer| io::copy(&mut reader, writer).map(drop))
    })
}

/// Downloads only the 4 m overview of every tile: the first part of each
/// file, which holds the header and the overview's pixels. That is about a
/// fifth of the data, which matters when the tiles cover a whole country.
/// A tile whose overview turns out to be unreadable is downloaded whole.
pub fn download_overviews(agent: &ureq::Agent, tiles: &[TileRef], dir: &Path, jobs: usize) -> Result<Vec<PathBuf>> {
    download_all(tiles, dir, "4m", jobs, |url, path| {
        let mut bytes = fetch_range(agent, url, 0, OVERVIEW_PREFIX_BYTES)?;
        let needed = geotiff::overview_prefix_len(&bytes)?;
        if needed > bytes.len() {
            let rest = fetch_range(agent, url, bytes.len(), needed)?;
            bytes.extend_from_slice(&rest);
            ensure!(bytes.len() == needed, "the server returned a short range");
        }
        bytes.truncate(needed);
        if !geotiff::overview_decodes(&bytes) {
            // The 2 m data of such a tile reads fine; `read_overview` uses it.
            println!("  the overview of {url} cannot be decoded, keeping the whole tile");
            bytes.clear();
            agent.get(url).call()?.into_reader().read_to_end(&mut bytes)?;
        }
        write_atomically(path, |writer| writer.write_all(&bytes))
    })
}

/// Requests bytes `start..end` of a file.
fn fetch_range(agent: &ureq::Agent, url: &str, start: usize, end: usize) -> Result<Vec<u8>> {
    let response = agent.get(url).set("Range", &format!("bytes={start}-{}", end - 1)).call()?;
    ensure!(response.status() == 206, "the server ignored the range request (status {})", response.status());
    let mut bytes = Vec::with_capacity(end - start);
    response.into_reader().take((end - start) as u64).read_to_end(&mut bytes)?;
    Ok(bytes)
}

/// Runs `fetch` for every tile that is not cached yet, on `jobs` threads.
fn download_all(
    tiles: &[TileRef],
    dir: &Path,
    suffix: &str,
    jobs: usize,
    fetch: impl Fn(&str, &Path) -> Result<()> + Sync,
) -> Result<Vec<PathBuf>> {
    fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let paths: Vec<PathBuf> = tiles.iter().map(|tile| dir.join(tile.file_name(suffix))).collect();
    let pending: Vec<usize> = (0..tiles.len()).filter(|&i| !paths[i].exists()).collect();
    println!("{} of {} tiles cached, downloading {}", tiles.len() - pending.len(), tiles.len(), pending.len());

    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let progress_step = (pending.len() / 20).max(100);
    let failures: Vec<anyhow::Error> = thread::scope(|scope| {
        let workers: Vec<_> = (0..jobs.clamp(1, pending.len().max(1)))
            .map(|_| {
                scope.spawn(|| {
                    let mut errors = Vec::new();
                    while let Some(&index) = pending.get(next.fetch_add(1, Ordering::Relaxed)) {
                        if let Err(error) = with_retries(&tiles[index].url, || fetch(&tiles[index].url, &paths[index]))
                        {
                            errors.push(error);
                        }
                        let finished = done.fetch_add(1, Ordering::Relaxed) + 1;
                        if finished.is_multiple_of(progress_step) || finished == pending.len() {
                            println!("  {finished} / {}", pending.len());
                        }
                    }
                    errors
                })
            })
            .collect();
        workers.into_iter().flat_map(|worker| worker.join().expect("download worker panicked")).collect()
    });

    let failed = failures.len();
    match failures.into_iter().next() {
        Some(error) => Err(error.context(format!("{failed} tiles failed to download; rerun to retry them"))),
        None => Ok(paths),
    }
}

fn with_retries(url: &str, attempt: impl Fn() -> Result<()>) -> Result<()> {
    let mut last_error = anyhow!("no attempt made");
    for round in 0..ATTEMPTS {
        if round > 0 {
            thread::sleep(Duration::from_secs(1 << round));
        }
        match attempt() {
            Ok(()) => return Ok(()),
            Err(error) => last_error = error,
        }
    }
    Err(last_error.context(format!("downloading {url}")))
}

/// Writes to a `.part` file first so a partial download never looks cached.
fn write_atomically(path: &Path, write: impl FnOnce(&mut BufWriter<File>) -> io::Result<()>) -> Result<()> {
    let partial = path.with_extension("part");
    let mut writer = BufWriter::new(File::create(&partial)?);
    write(&mut writer)?;
    writer.into_inner().map_err(|error| error.into_error())?.sync_all()?;
    fs::rename(&partial, path)?;
    Ok(())
}
