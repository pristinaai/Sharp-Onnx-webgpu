import { cp, mkdir } from 'node:fs/promises'
import { resolve } from 'node:path'

const root = process.cwd()
const srcDir = resolve(root, 'node_modules/@playcanvas/supersplat-viewer/public')
const dstDir = resolve(root, 'public/supersplat-viewer')

await mkdir(dstDir, { recursive: true })
for (const file of ['index.html', 'index.css', 'index.js']) {
  await cp(resolve(srcDir, file), resolve(dstDir, file))
}

console.log(`Copied SuperSplat viewer assets to ${dstDir}`)
