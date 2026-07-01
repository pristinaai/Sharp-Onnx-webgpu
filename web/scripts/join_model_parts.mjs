#!/usr/bin/env node
/**
 * Reassemble split .onnx.data sidecars after git clone / git lfs pull.
 */
import { open, readFile, stat, access, unlink, readdir } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const modelsDir = path.join(webRoot, 'public/models')

async function exists(filePath) {
  try {
    await access(filePath)
    return true
  } catch {
    return false
  }
}

async function joinFromManifest(manifestPath) {
  const manifest = JSON.parse(await fsRead(manifestPath))
  const outPath = path.join(modelsDir, manifest.file)

  if (await exists(outPath)) {
    const existing = await stat(outPath)
    if (existing.size === manifest.size) {
      return false
    }
    console.warn(`join_model_parts: removing incomplete ${manifest.file} (${existing.size} vs ${manifest.size})`)
    await unlink(outPath)
  }

  const handle = await open(outPath, 'w')
  try {
    for (const partName of manifest.parts) {
      const partPath = path.join(modelsDir, partName)
      if (!(await exists(partPath))) {
        throw new Error(`Missing part ${partName} — run git lfs pull`)
      }
      await handle.write(await readFile(partPath))
    }
  } finally {
    await handle.close()
  }

  const joined = await stat(outPath)
  if (joined.size !== manifest.size) {
    await unlink(outPath).catch(() => {})
    throw new Error(`Joined ${manifest.file} size mismatch: ${joined.size} vs ${manifest.size}`)
  }

  console.log(`join_model_parts: assembled ${manifest.file} (${joined.size} bytes)`)
  return true
}

function fsRead(filePath) {
  return readFile(filePath, 'utf8')
}

async function main() {
  const entries = await readdir(modelsDir)
  const manifests = entries.filter((name) => name.endsWith('.onnx.data.manifest.json'))
  if (manifests.length === 0) {
    return
  }

  for (const manifestName of manifests) {
    await joinFromManifest(path.join(modelsDir, manifestName))
  }
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
