((root) => {
  function findDirectionalNeighbor(tiles, current, key) {
    const direction = {
      ArrowLeft: [-1, 0], ArrowRight: [1, 0],
      ArrowUp: [0, -1], ArrowDown: [0, 1],
    }[key];
    if (!direction || !Array.isArray(tiles) || !current?.dataset) return null;
    const col = Number(current.dataset.gridCol);
    const row = Number(current.dataset.gridRow);
    if (!Number.isInteger(col) || !Number.isInteger(row)) return null;
    const [dx, dy] = direction;
    return tiles.filter((tile) => tile !== current && tile?.dataset)
      .map((tile) => ({
        tile,
        col: Number(tile.dataset.gridCol),
        row: Number(tile.dataset.gridRow),
      }))
      .filter((item) => Number.isInteger(item.col) && Number.isInteger(item.row)
        && (dx ? (item.col - col) * dx > 0 : (item.row - row) * dy > 0))
      .sort((a, b) => {
        const primary = (item) => dx ? Math.abs(item.col - col) : Math.abs(item.row - row);
        const secondary = (item) => dx ? Math.abs(item.row - row) : Math.abs(item.col - col);
        return primary(a) + 10 * secondary(a) - primary(b) - 10 * secondary(b)
          || a.row - b.row || a.col - b.col;
      })[0]?.tile ?? null;
  }

  const api = { findDirectionalNeighbor };
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.PramaanQuilt = api;
})(globalThis);
