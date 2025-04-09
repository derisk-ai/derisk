
## 👋 Introduction

***DeRisk-Web*** is an **Open source chat UI** for [**DERISK**](https://github.com/derisk-ai/derisk).
Also, it is a **LLM to Vision** solution. 


## 💪🏻 Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) >= 16
- [npm](https://npmjs.com/) >= 8
- [yarn](https://yarnpkg.com/) >= 1.22
- Supported OSes: Linux, macOS and Windows

### Installation

```sh
# Install dependencies
npm install
yarn install
```

### Usage
```sh
cp .env.template .env
```
edit the `API_BASE_URL` to the real address

```sh
# development model
npm run dev
yarn dev
```

## 🚀 Use In DB-GPT

```sh
npm run compile
yarn compile

# copy compile file to DeRisk static file dictory
cp -rf out/* ../derisk/app/static 

```

## 📚 Documentation

For full documentation, visit [document](https://docs.dbgpt.site/).


## Usage
  [gpt-vis](https://github.com/antvis/GPT-Vis) for markdown support.
  [ant-design](https://github.com/ant-design/ant-design) for ui components.
  [next.js](https://github.com/vercel/next.js) for server side rendering.
  [@antv/g2](https://github.com/antvis/g2#readme) for charts.

## License

DeRisk-Web is licensed under the [MIT License](LICENSE).

---

🌟 If you find it helpful, don't forget to give it a star on GitHub! Stars are like little virtual hugs that keep us going! We appreciate every single one we receive.


Happy coding! 😊

