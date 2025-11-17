#!/usr/bin/env node
/**
 * 构建脚本：将 src 目录编译到 dist 目录
 * 
 * 功能：
 * 1. 清理 dist 目录
 * 2. 复制 HTML、CSS 和 models 文件
 * 3. 编译 TypeScript 文件
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const SRC_DIR = path.join(__dirname, 'src');
const DIST_DIR = path.join(__dirname, 'dist');

// 清理 dist 目录
function clean() {
  console.log('🧹 清理 dist 目录...');
  if (fs.existsSync(DIST_DIR)) {
    fs.rmSync(DIST_DIR, { recursive: true, force: true });
  }
  fs.mkdirSync(DIST_DIR, { recursive: true });
}

// 复制文件
function copyFiles() {
  console.log('📋 复制文件...');
  
  // 复制 HTML 文件
  const htmlSrc = path.join(SRC_DIR, 'index.html');
  const htmlDist = path.join(DIST_DIR, 'index.html');
  fs.copyFileSync(htmlSrc, htmlDist);
  console.log('  ✓ index.html');
  
  // 复制 CSS 文件
  const cssSrc = path.join(SRC_DIR, 'style.css');
  const cssDist = path.join(DIST_DIR, 'style.css');
  fs.copyFileSync(cssSrc, cssDist);
  console.log('  ✓ style.css');
  
  // 复制 models 目录（如果存在）
  const modelsSrc = path.join(SRC_DIR, 'models');
  const modelsDist = path.join(DIST_DIR, 'models');
  if (fs.existsSync(modelsSrc)) {
    // 递归复制整个目录
    copyRecursiveSync(modelsSrc, modelsDist);
    console.log('  ✓ models/');
  } else {
    console.log('  ℹ️  src/models/ 不存在，跳过模型文件复制');
  }
}

// 递归复制目录
function copyRecursiveSync(src, dest) {
  const exists = fs.existsSync(src);
  const stats = exists && fs.statSync(src);
  const isDirectory = exists && stats.isDirectory();
  
  if (isDirectory) {
    if (!fs.existsSync(dest)) {
      fs.mkdirSync(dest, { recursive: true });
    }
    fs.readdirSync(src).forEach(childItemName => {
      copyRecursiveSync(
        path.join(src, childItemName),
        path.join(dest, childItemName)
      );
    });
  } else {
    fs.copyFileSync(src, dest);
  }
}

// 编译 TypeScript
function compileTypeScript() {
  console.log('🔨 编译 TypeScript...');
  try {
    execSync('npx tsc', { stdio: 'inherit' });
    console.log('  ✓ script.ts -> script.js');
  } catch (error) {
    console.error('❌ TypeScript 编译失败');
    process.exit(1);
  }
}

// 主函数
function build() {
  console.log('🚀 开始构建...\n');
  
  clean();
  copyFiles();
  compileTypeScript();
  
  console.log('\n✅ 构建完成！');
  console.log(`📁 输出目录: ${DIST_DIR}`);
}

// 运行构建
build();

