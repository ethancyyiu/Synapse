// Checks that every local link, script, image, and data file referenced by the pages and main documents exists.
// External (http, mailto, data) links are not fetched. Exits 1 if anything is missing.
// node tools/check_links.mjs
import { readFile, access } from 'node:fs/promises';

const root = new URL('../', import.meta.url);
const files = ['index.html', 'archive.html', 'legacy.html', 'rhythm.html', 'fraud.html', 'README.md',
  'docs/pitch-script.md', 'docs/demo-guide.md', 'docs/research-story.md', 'docs/earlier-rps-experiment.md',
  'docs/synapse-fraud-plan.md', 'docs/synapse-fraud-protocol.md', 'docs/synapse-fraud-research.md',
  'docs/synapse-fraud-demo-guide.md'];
const missing = [];
let checked = 0;
for (const file of files) {
  const body = await readFile(new URL(file, root), 'utf8');
  const found = file.endsWith('.md')
    ? [...body.matchAll(/\]\(([^)\s]+)\)/g)].map(m => m[1])
    : [...body.matchAll(/(?:href|src)="([^"]+)"/g)].map(m => m[1]);
  for (const link of new Set(found)) {
    if (/^(?:[a-z]+:|#|\/\/)/i.test(link)) continue;
    const target = new URL(link.split('#')[0].split('?')[0], new URL(file, root));
    checked++;
    try { await access(target); } catch { missing.push(`${file} -> ${link}`); }
  }
}
if (missing.length) { console.error(missing.map(m => 'MISSING  ' + m).join('\n')); process.exit(1); }
console.log(`links: ${checked} local references in ${files.length} files all exist`);
