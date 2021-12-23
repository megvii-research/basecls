import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import ModelIndex from '../components/ModelIndex.vue'
import ModelDetail from '../components/ModelDetail.vue'
import store from '../store'
import VChart from 'vue-echarts'

function zooIndex () {
  const app = createApp(ModelIndex)

  app.component('v-chart', VChart)
  app.use(ElementPlus).use(store)
  app.mount("#model-index")
}

function zooDetail () {
  if (document.getElementById('modelInfoContainer') !== null) {
    const app = createApp(ModelDetail)
    app.use(ElementPlus).use(store)
    app.mount('#modelInfoContainer')
  }
}

const meta = document.getElementById('sphinx-pagename')
if (meta === null) {
  console.log('No sphinx pagename found, ignore')
}
const pagename = meta.attributes.pagename.value
console.log(`Sphinx pagename=${pagename}`)

if (pagename === 'zoo/index') {
  zooIndex()
} else if (pagename.startsWith('zoo/')) {
  zooDetail()
}
