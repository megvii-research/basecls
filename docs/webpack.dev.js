const path = require('path');
const { merge } = require("webpack-merge");
const exec = require('child_process').exec;
const WatchPlugin = require('webpack-watch-files-plugin').default;
const ShellPlugin = require('webpack-shell-plugin-next');
const common = require('./webpack.common.js');

module.exports = merge(common, {
  mode: 'development',
  devServer: {
    contentBase: path.join(__dirname, 'build/html'),
    watchContentBase: true,
    compress: false,
    port: 1919,
    hot: false,
    liveReload: true,
    publicPath: '/_static/',
    writeToDisk: true,

    // Ignore host check to develop behind brainpp proxy
    host: "0.0.0.0",
    disableHostCheck: true ,
    proxy: {
      '/json': {
        target: 'http://localhost:3010',
      },
    },
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, PATCH, OPTIONS",
      "Access-Control-Allow-Headers": "X-Requested-With, content-type, Authorization"
    }
  },
  plugins: [
    new WatchPlugin({
      files: [
        './source/**/*.rst',
        './source/**/*.json',
        './frontend/**/*.*',
        // watching the generated macros causes vicious cycles
        '!./source/_templates/*.html',
      ],
    }),
    new ShellPlugin({
      onBuildEnd: {
        scripts: ['make html'],
      },
      // dev=false here to force every build to trigger make, the default is
      // first build only.
      dev: false
    }),
  ],
});
