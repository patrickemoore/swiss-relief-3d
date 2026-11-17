// Builds the gazetteer the map labels come from: peaks, lakes, glaciers and
// towns of Switzerland, from Wikidata, with Swiss LV95 coordinates and a
// score that says how important each is. Writes crates/render/assets/labels.json,
// which the viewer embeds.
//
//     node tools/labels.mjs
//
// Peaks are scored by how well known they are (the number of Wikipedias
// that have an article), their prominence and their height; lakes by area,
// glaciers by length, towns by population. The viewer shows the best-scored
// first and the rest as the view comes closer.
import { writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const OUT = resolve(dirname(fileURLToPath(import.meta.url)), '../crates/render/assets/labels.json');
const ENDPOINT = 'https://query.wikidata.org/sparql';
const LANGUAGES = 'en,de,fr,it,rm';
// Fame is counted in articles on these Wikipedias: a count of all sitelinks
// would be swamped by the ones that have an article for every hill.
const MAJOR_WIKIS = ['en', 'de', 'fr', 'it', 'es', 'pl', 'ru', 'ja', 'zh', 'pt']
  .map((code) => `<https://${code}.wikipedia.org/>`)
  .join(' ');
const FAME = `OPTIONAL { VALUES ?wiki { ${MAJOR_WIKIS} } ?article schema:about ?item; schema:isPartOf ?wiki. }`;

const QUERIES = {
  peak: `
    SELECT ?item ?label ?coord ?elevation ?prominence (COUNT(DISTINCT ?article) AS ?fame) WHERE {
      ?item wdt:P31 wd:Q8502; wdt:P17 wd:Q39; wdt:P625 ?coord; wdt:P2044 ?elevation.
      OPTIONAL { ?item wdt:P2660 ?prominence. }
      FILTER(?elevation >= 1200)
      ${FAME}
      SERVICE wikibase:label { bd:serviceParam wikibase:language "${LANGUAGES}". ?item rdfs:label ?label. }
    } GROUP BY ?item ?label ?coord ?elevation ?prominence`,
  lake: `
    SELECT ?item ?label ?coord ?area ?part (COUNT(DISTINCT ?article) AS ?fame) WHERE {
      ?item wdt:P31/wdt:P279* wd:Q23397; wdt:P17 wd:Q39; wdt:P625 ?coord.
      ?item p:P2046/psn:P2046/wikibase:quantityAmount ?area.
      OPTIONAL { ?item wdt:P361 ?part. }
      FILTER(?area >= 250000)
      ${FAME}
      SERVICE wikibase:label { bd:serviceParam wikibase:language "${LANGUAGES}". ?item rdfs:label ?label. }
    } GROUP BY ?item ?label ?coord ?area ?part`,
  glacier: `
    SELECT ?item ?label ?coord ?length (COUNT(DISTINCT ?article) AS ?fame) WHERE {
      ?item wdt:P31/wdt:P279* wd:Q35666; wdt:P625 ?coord.
      FILTER(geof:latitude(?coord) > 45.8 && geof:latitude(?coord) < 47.9 && geof:longitude(?coord) > 5.9 && geof:longitude(?coord) < 10.6)
      OPTIONAL { ?item p:P2043/psn:P2043/wikibase:quantityAmount ?length. }
      ${FAME}
      SERVICE wikibase:label { bd:serviceParam wikibase:language "${LANGUAGES}". ?item rdfs:label ?label. }
    } GROUP BY ?item ?label ?coord ?length`,
  town: `
    SELECT ?item ?label ?coord (MAX(?p) AS ?population) (COUNT(DISTINCT ?article) AS ?fame) WHERE {
      ?item wdt:P31 wd:Q70208; wdt:P625 ?coord; wdt:P1082 ?p.
      ${FAME}
      SERVICE wikibase:label { bd:serviceParam wikibase:language "${LANGUAGES}". ?item rdfs:label ?label. }
    } GROUP BY ?item ?label ?coord HAVING(MAX(?p) >= 1500)`,
};

/**
 * Names every Swiss map shows, whatever the numbers say: their score is
 * raised so that they come first.
 */
const STARS = {
  Matterhorn: 120, Jungfrau: 110, Eiger: 100, 'Mount Pilatus': 95, Rigi: 95, Säntis: 95, 'Piz Bernina': 90, Titlis: 85,
  Mönch: 70, Dufourspitze: 70, Tödi: 65, Niesen: 60, Finsteraarhorn: 55, Dom: 40, Weisshorn: 40,
  'Aletsch Glacier': 100, 'Rhone Glacier': 60,
};

/** Figures Wikidata has wrong, in square kilometres. */
const LAKE_AREA_KM2 = { Gelmersee: 0.64 };

/** How many of each kind to keep, best first. */
// Glaciers are found by position, so some are over the border and go when
// the viewer keeps only what its terrain covers.
const KEEP = { peak: 450, lake: 90, glacier: 45, town: 420 };

async function query(sparql) {
  const response = await fetch(`${ENDPOINT}?format=json&query=${encodeURIComponent(sparql)}`, {
    headers: { 'User-Agent': 'swiss-relief-3d gazetteer build' },
  });
  if (!response.ok) throw new Error(`Wikidata: HTTP ${response.status} ${await response.text()}`);
  const { results } = await response.json();
  return results.bindings.map((row) => Object.fromEntries(Object.entries(row).map(([key, cell]) => [key, cell.value])));
}

/** WGS84 to LV95 by swisstopo's approximate formulas, good to a metre. */
function lv95(latitude, longitude) {
  const phi = (latitude * 3600 - 169028.66) / 10000;
  const lambda = (longitude * 3600 - 26782.5) / 10000;
  const east =
    2600072.37 + 211455.93 * lambda - 10938.51 * lambda * phi - 0.36 * lambda * phi * phi - 44.54 * lambda ** 3;
  const north =
    1200147.07 + 308807.95 * phi + 3745.25 * lambda * lambda + 76.63 * phi * phi - 194.56 * lambda * lambda * phi + 119.79 * phi ** 3;
  return [Math.round(east), Math.round(north)];
}

function point(wkt) {
  const match = /Point\(([-\d.]+) ([-\d.]+)\)/.exec(wkt);
  return match ? [Number(match[2]), Number(match[1])] : null;
}

const number = (value) => (value === undefined ? undefined : Number(value));

function score(kind, row) {
  const fame = number(row.fame) ?? 0;
  switch (kind) {
    case 'peak': {
      const prominence = Math.min(number(row.prominence) ?? 0, 1500);
      return fame * 12 + prominence / 40 + Math.max(0, number(row.elevation) - 3000) / 100;
    }
    case 'lake':
      return Math.sqrt(LAKE_AREA_KM2[row.label] ?? number(row.area) / 1e6) * 10 + fame * 2;
    case 'glacier':
      return (number(row.length) ?? 0) / 500 + fame * 4;
    case 'town':
      // Size, and the renown that makes a village of 5,000 a place on the map.
      return Math.sqrt(number(row.population)) / 10 + fame * 2.5;
  }
}

/** Items within a kilometre of a better one of the same kind are the same thing twice. */
function dedupe(labels) {
  const kept = [];
  for (const label of labels) {
    const near = kept.some(
      (other) => other.kind === label.kind && Math.hypot(other.east - label.east, other.north - label.north) < 1000,
    );
    if (!near) kept.push(label);
  }
  return kept;
}

const labels = [];
for (const [kind, sparql] of Object.entries(QUERIES)) {
  let rows = await query(sparql);
  // A part of a lake in the list, such as the Obersee of Lake Constance,
  // is not a lake of its own.
  const items = new Set(rows.map((row) => row.item));
  rows = rows.filter((row) => !(row.part && items.has(row.part)));
  const seen = new Set();
  const scored = [];
  for (const row of rows) {
    const coordinates = point(row.coord);
    if (!coordinates || seen.has(row.item) || /^Q\d+$/.test(row.label)) continue;
    seen.add(row.item);
    const [east, north] = lv95(...coordinates);
    const stardom = STARS[row.label] ?? 0;
    const label = { kind, name: row.label, east, north, score: Math.round((score(kind, row) + stardom) * 10) / 10 };
    if (kind === 'peak') label.elevation = Math.round(number(row.elevation));
    scored.push(label);
  }
  scored.sort((a, b) => b.score - a.score);
  const kept = dedupe(scored).slice(0, KEEP[kind]);
  console.log(`${kind}: ${rows.length} rows, kept ${kept.length}; best: ${kept.slice(0, 8).map((l) => l.name).join(', ')}`);
  labels.push(...kept);
}

writeFileSync(OUT, JSON.stringify(labels, null, 0).replaceAll('},{', '},\n{') + '\n');
console.log(`${labels.length} labels -> ${OUT}`);
