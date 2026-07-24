import { readFileSync, writeFileSync, mkdirSync, readdirSync } from 'node:fs';
import { join, relative } from 'node:path';
import { exit } from 'node:process';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const REPO_ROOT = join(import.meta.dirname, '..', '..');
const PLUGINS_DIR = join(REPO_ROOT, 'src', 'plugins');
const OUTPUT_PATH = join(REPO_ROOT, 'docs', 'data', 'stack-data.json');
const SCHEMA_PATH = join(REPO_ROOT, 'docs', 'data', 'stack-data.schema.json');

// ---------------------------------------------------------------------------
// Memory type mapping  (C++ enum -> human-readable)
// ---------------------------------------------------------------------------

const MEM_TYPE_MAP = {
  'DRAM_SEG': 'DRAM',
  'VRAM_SEG': 'VRAM',
  'FILE_SEG': 'FILE',
  'OBJ_SEG':  'OBJ',
  'BLK_SEG':  'BLK',
};

// ---------------------------------------------------------------------------
// Category mapping  (plugin name -> diagram grouping)
// ---------------------------------------------------------------------------

const CATEGORY_MAP = {
  'UCX':        'network',
  'Libfabric':  'network',
  'MOONCAKE':   'network',
  'DOCA GPUNetIO': 'gpu-direct',
  'UCCL':       'network',
  'GDS':        'storage',
  'GDS_MT':     'storage',
  'POSIX':      'storage',
  'HF3FS':      'storage',
  'OBJ':        'object-storage',
  'AZURE_BLOB': 'object-storage',
  'GUSLI':      'block-storage',
};

// ---------------------------------------------------------------------------
// Plugin discovery
// ---------------------------------------------------------------------------

function discoverPluginFiles() {
  const entries = readdirSync(PLUGINS_DIR, { withFileTypes: true });
  const pluginFiles = [];

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    if (entry.name === 'telemetry') continue; // Different interface

    const dirPath = join(PLUGINS_DIR, entry.name);
    const files = readdirSync(dirPath).filter(f => f.endsWith('_plugin.cpp'));

    for (const file of files) {
      pluginFiles.push(join(dirPath, file));
    }
  }

  return pluginFiles;
}

// ---------------------------------------------------------------------------
// Primary parser: template create() pattern
// ---------------------------------------------------------------------------

function parseTemplateCreate(content) {
  // Locate nixl_plugin_init() function body
  const initIdx = content.indexOf('nixl_plugin_init()');
  if (initIdx === -1) return null;
  const body = content.slice(initIdx);

  // Extract name and version from create() call
  const nameVersionRe = /create\s*\(\s*NIXL_PLUGIN_API_VERSION\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"/s;
  const nvMatch = body.match(nameVersionRe);
  if (!nvMatch) return null;

  const name = nvMatch[1];
  const version = nvMatch[2];

  // Try inline brace list for memory types: {DRAM_SEG, VRAM_SEG, ...});
  const memListInlineRe = /\{\s*((?:[A-Z_]+_SEG)(?:\s*,\s*[A-Z_]+_SEG)*)\s*\}\s*\)\s*;/s;
  const memMatch = body.match(memListInlineRe);

  let memoryTypes;
  if (memMatch) {
    const rawSegs = memMatch[1].split(/\s*,\s*/);
    memoryTypes = rawSegs.map(seg => MEM_TYPE_MAP[seg.trim()]).filter(Boolean);
  } else {
    // Fallback: variable reference (OBJ plugin pattern)
    const varRefRe = /nixl_mem_list_t\s+\w+\s*=\s*\{([^}]+)\}/;
    const varMatch = content.match(varRefRe);
    if (!varMatch) return null;

    const rawSegs = varMatch[1].split(/\s*,\s*/);
    memoryTypes = rawSegs.map(seg => MEM_TYPE_MAP[seg.trim()]).filter(Boolean);
  }

  if (memoryTypes.length === 0) return null;

  return { name, version, memoryTypes };
}

// ---------------------------------------------------------------------------
// Fallback parser: manual struct pattern (DOCA GPUNetIO)
// ---------------------------------------------------------------------------

function parseManualStruct(content) {
  const pluginNameRe = /static\s+const\s+char\s*\*\s*PLUGIN_NAME\s*=\s*"([^"]+)"/;
  const pluginVersionRe = /static\s+const\s+char\s*\*\s*PLUGIN_VERSION\s*=\s*"([^"]+)"/;

  const nameMatch = content.match(pluginNameRe);
  const versionMatch = content.match(pluginVersionRe);
  if (!nameMatch || !versionMatch) return null;

  const name = nameMatch[1];
  const version = versionMatch[1];

  // Extract memory types from push_back calls
  const pushBackRe = /\.push_back\s*\(\s*([A-Z_]+_SEG)\s*\)/g;
  const memoryTypes = [];
  let m;
  while ((m = pushBackRe.exec(content)) !== null) {
    const mapped = MEM_TYPE_MAP[m[1]];
    if (mapped) memoryTypes.push(mapped);
  }

  if (memoryTypes.length === 0) return null;

  return { name, version, memoryTypes };
}

// ---------------------------------------------------------------------------
// Orchestrator: parse a single plugin file
// ---------------------------------------------------------------------------

function parsePluginFile(filePath) {
  const content = readFileSync(filePath, 'utf-8');

  if (!content.includes('nixl_plugin_init')) {
    return { type: 'not-backend', path: filePath };
  }

  // Try primary pattern
  const result = parseTemplateCreate(content);
  if (result) {
    return { type: 'success', ...result, sourceFile: relative(REPO_ROOT, filePath) };
  }

  // Try fallback pattern
  const fallback = parseManualStruct(content);
  if (fallback) {
    return { type: 'success', ...fallback, sourceFile: relative(REPO_ROOT, filePath) };
  }

  return {
    type: 'error',
    path: filePath,
    reason: 'Contains nixl_plugin_init but no parseable pattern found',
  };
}

// ---------------------------------------------------------------------------
// Minimal schema validator
// ---------------------------------------------------------------------------

function validateAgainstSchema(data, schema) {
  const errors = [];

  // 1. Root is an array
  if (!Array.isArray(data)) {
    errors.push('Root must be an array');
    return { valid: false, errors };
  }

  const itemSchema = schema.items;
  const requiredFields = itemSchema.required || [];
  const allowedProps = Object.keys(itemSchema.properties || {});
  const memTypeEnum = itemSchema.properties.memoryTypes.items.enum;
  const categoryEnum = itemSchema.properties.category.enum;
  const versionPattern = new RegExp(itemSchema.properties.version.pattern);
  const minMemItems = itemSchema.properties.memoryTypes.minItems || 0;

  for (let i = 0; i < data.length; i++) {
    const item = data[i];
    const prefix = `[${i}]`;

    // 2. Each item is an object
    if (typeof item !== 'object' || item === null || Array.isArray(item)) {
      errors.push(`${prefix} must be an object`);
      continue;
    }

    // 3. Required fields
    for (const field of requiredFields) {
      if (!(field in item)) {
        errors.push(`${prefix} missing required field "${field}"`);
      }
    }

    // 9. No additional properties
    for (const key of Object.keys(item)) {
      if (!allowedProps.includes(key)) {
        errors.push(`${prefix} has additional property "${key}"`);
      }
    }

    // 4. name is a string
    if ('name' in item && typeof item.name !== 'string') {
      errors.push(`${prefix}.name must be a string`);
    }

    // 5. version is a string matching pattern
    if ('version' in item) {
      if (typeof item.version !== 'string') {
        errors.push(`${prefix}.version must be a string`);
      } else if (!versionPattern.test(item.version)) {
        errors.push(`${prefix}.version "${item.version}" does not match pattern ${itemSchema.properties.version.pattern}`);
      }
    }

    // 6. memoryTypes is a non-empty array of valid enum strings
    if ('memoryTypes' in item) {
      if (!Array.isArray(item.memoryTypes)) {
        errors.push(`${prefix}.memoryTypes must be an array`);
      } else {
        if (item.memoryTypes.length < minMemItems) {
          errors.push(`${prefix}.memoryTypes must have at least ${minMemItems} item(s)`);
        }
        for (let j = 0; j < item.memoryTypes.length; j++) {
          const mt = item.memoryTypes[j];
          if (typeof mt !== 'string') {
            errors.push(`${prefix}.memoryTypes[${j}] must be a string`);
          } else if (!memTypeEnum.includes(mt)) {
            errors.push(`${prefix}.memoryTypes[${j}] "${mt}" is not in enum [${memTypeEnum.join(', ')}]`);
          }
        }
      }
    }

    // 7. sourceFile is a string
    if ('sourceFile' in item && typeof item.sourceFile !== 'string') {
      errors.push(`${prefix}.sourceFile must be a string`);
    }

    // 8. category is a string in enum
    if ('category' in item) {
      if (typeof item.category !== 'string') {
        errors.push(`${prefix}.category must be a string`);
      } else if (!categoryEnum.includes(item.category)) {
        errors.push(`${prefix}.category "${item.category}" is not in enum [${categoryEnum.join(', ')}]`);
      }
    }
  }

  return { valid: errors.length === 0, errors };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

function main() {
  // 1. Discover plugin files
  const pluginFiles = discoverPluginFiles();

  // 2. Parse each file
  const results = pluginFiles.map(parsePluginFile);

  // 3. Separate results
  const successes = results.filter(r => r.type === 'success');
  const notBackend = results.filter(r => r.type === 'not-backend');
  const errors = results.filter(r => r.type === 'error');

  // 4. Print warnings for non-backend files
  for (const r of notBackend) {
    console.warn(`Warning: ${r.path} does not contain nixl_plugin_init, skipping`);
  }

  // 5. Print errors
  for (const r of errors) {
    console.error(`Error: ${r.path} -- ${r.reason}`);
  }

  // 6. Exit non-zero if any errors
  if (errors.length > 0) {
    console.error(`\n${errors.length} plugin(s) failed to parse.`);
    exit(1);
  }

  // 7. Assign categories
  const data = successes.map(r => {
    const category = CATEGORY_MAP[r.name];
    if (!category) {
      console.warn(`Warning: No category mapping for plugin '${r.name}', using 'other'`);
    }
    return {
      name: r.name,
      version: r.version,
      memoryTypes: r.memoryTypes,
      sourceFile: r.sourceFile,
      category: category || 'other',
    };
  });

  // 9. Sort alphabetically by name
  data.sort((a, b) => a.name.localeCompare(b.name));

  // 10. Load schema and validate
  const schema = JSON.parse(readFileSync(SCHEMA_PATH, 'utf-8'));
  const validation = validateAgainstSchema(data, schema);

  // 11. Exit non-zero if validation fails
  if (!validation.valid) {
    console.error('Schema validation failed:');
    for (const err of validation.errors) {
      console.error(`  - ${err}`);
    }
    exit(1);
  }

  // 12. Ensure output directory exists
  mkdirSync(join(REPO_ROOT, 'docs', 'data'), { recursive: true });

  // 13. Write output
  writeFileSync(OUTPUT_PATH, JSON.stringify(data, null, 2) + '\n');

  // 14. Print summary
  console.log(`Generated stack-data.json: ${data.length} plugins parsed, written to docs/data/stack-data.json`);
  exit(0);
}

main();
