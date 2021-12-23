# Basecls Documentation

## Basic Architecture

```plaintext
+---------------------------------+
|          SphinxDoc              |
|                    +------------+
|                    | _static    |    yarn build  +--------------+
|                    | _templates |<---------------+   frontend   |
+---------------+----+------------+                +--------------+
                |
                | make html
                v
           Final HTML
```

frontend use webpack to build static files and put generated files into `_static` and `_templates`, then sphinx build whole document.

## How to build document

```bash
make html
```

## How to develop frontend files

Notes: Before you commit, static file located in `source/_static/` and `source/_templates` should be generated and commit in. This enables other developers to build whole document without frontend toolkit. Any merge conflicts about these files could be ignored. See build command in **Build Production Files** section.

### Install requirements

#### Step1: Install nodeJS dependencies

```bash
npm install -g webpack-cli yarn
yarn
```

#### Step2: Setup Brain++ OSS credentials

Make sure `aws s3 ls basecls/json/` return subdirectory properly.

### Build Production Files

```bash
yarn build:production
```

### Development Guide

#### Step1: Start OSS Proxy

Execute `python3 s3_proxy.py` to start a s3 proxy server.

#### Step2: Start Webpack debug server

Use `yarn build:dev` to start a develop server, this server do several things:

* Start a develop server listen at 1919
* call `make html` to generate Sphinx pages when file changes
* Serve files in `_static` from webpack and Sphinx files from `build/html`

#### Step3: Happy Coding

* Visit `:1919` to preview result
* Update any files and wait webpack server finish, then reload page to see updated result
