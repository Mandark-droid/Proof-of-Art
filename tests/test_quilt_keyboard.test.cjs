const assert = require('node:assert/strict');
const { test } = require('node:test');
const { findDirectionalNeighbor } = require('../pramaan/ui/assets/quilt.js');

const tile = (id, col, row) => ({ id, dataset: { gridCol: String(col), gridRow: String(row) } });

test('arrow movement prefers the nearest tile on the same row or column', () => {
  const current = tile('current', 3, 4);
  const sameRow = tile('same-row', 1, 4);
  const diagonal = tile('diagonal', 2, 3);
  assert.equal(findDirectionalNeighbor([current, sameRow, diagonal], current, 'ArrowLeft'), sameRow);
});

test('arrow movement can cross geographic gaps and returns no tile at the edge', () => {
  const current = tile('current', 3, 4);
  const diagonal = tile('diagonal', 2, 3);
  const lower = tile('lower', 4, 7);
  assert.equal(findDirectionalNeighbor([current, diagonal, lower], current, 'ArrowLeft'), diagonal);
  assert.equal(findDirectionalNeighbor([current, diagonal, lower], current, 'ArrowUp'), diagonal);
  const top = tile('top', 3, 0);
  assert.equal(findDirectionalNeighbor([top, current, diagonal, lower], top, 'ArrowUp'), null);
  assert.equal(findDirectionalNeighbor([current], current, 'Enter'), null);
});

test('unknown grid coordinates are ignored safely', () => {
  const current = tile('current', 3, 4);
  const unknown = { id: 'unknown', dataset: {} };
  assert.equal(findDirectionalNeighbor([unknown, current], current, 'ArrowLeft'), null);
});
