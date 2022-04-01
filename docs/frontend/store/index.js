import { createStore } from "vuex";

export default createStore({
    getters: {
      APIBase: () => process.env.NODE_ENV !== 'production'?'':'https://www.megengine.org.cn/meta/basecls',
      OSSPath: () => 'https://data.megengine.org.cn/research/basecls/models'
    }
})
