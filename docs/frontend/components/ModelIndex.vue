<template>
    <div id="vue-container">
        <div id="chartOptions">
            <el-form :inline="true">
            <el-form-item>
                <el-switch v-model="chart_type" active-text="散点图" inactive-text="平行坐标图" active-value="scatter" inactive-value="parallel">
                </el-switch>
            </el-form-item>

            <el-form-item>
                <el-switch v-model="scatter_flops_log" active-text="logspace">
                </el-switch>
            </el-form-item>

            <el-form-item id="axis-selector" label="横轴数据源" :style="'display:'+((chart_type === 'scatter')?'inline-flex':'none')">
                <el-select v-model="axisDataSelect" size="mini">
                <el-option
                    v-for="item in axisDataOptions"
                    :key="item"
                    :label="dataDimName[item].display"
                    :value="item">
                </el-option>
                </el-select>
            </el-form-item>

            </el-form>
        </div>
        <v-chart ref="chart" v-on:click="redirectUrl" :option="option" id="chartContainer">
        </v-chart>
    </div>
</template>

<script>
import 'element-plus/dist/index.css'
import { use } from 'echarts/core'
import { SVGRenderer } from 'echarts/renderers'
import { ScatterChart, ParallelChart } from 'echarts/charts'
import {
  TooltipComponent,
  LegendComponent,
  ToolboxComponent
} from 'echarts/components'
import byteSize from 'byte-size'

use([
  SVGRenderer,
  TooltipComponent,
  LegendComponent,
  ScatterChart,
  ParallelChart,
  ToolboxComponent
])

// construct a formatter for label on axis
function axisFormatterConstructor (precision = 1) {
  return (value, _) => {
    const byteUnit = byteSize(value, { units: 'metric', precision: precision })
    return byteUnit.value + ' ' + byteUnit.unit.slice(0, byteUnit.unit.length - 1)
  }
}

/* map from raw json to series data
 * Key: key in json
 * Value:
 *   name: same as key
 *   display: name to display
 *   dim: index in data array. Must be unique.
*/
const dataDimName = {
  acc1: {
    name: 'acc1',
    display: 'Acc@1',
    dim: 0
  },
  acc5: {
    name: 'acc5',
    display: 'Acc@5',
    dim: 1
  },
  activations: {
    name: 'activations',
    display: 'Activations',
    dim: 2
  },
  flops: {
    name: 'flops',
    display: 'Flops',
    dim: 3
  },
  params: {
    name: 'params',
    display: 'Params',
    dim: 4
  },
  name: {
    name: 'name',
    display: 'Name',
    dim: 5
  },
  url: {
    name: 'url',
    display: 'Url',
    dim: 6
  }
}

// echarts common options
const commonOptions = {
  series: [],
  legend: {
    type: 'plain',
    show: true,
    top: 'bottom'
  },
  tooltip: {
    trigger: 'item',
    triggerOn: 'mousemove',
    position: 'top',
    formatter: (params) => {
      return `<b>${params.data[dataDimName.name.dim]}</b><br/>
       ${dataDimName.acc1.display}: ${params.data[dataDimName.acc1.dim]}% <br/>
       ${dataDimName.acc5.display}: ${params.data[dataDimName.acc5.dim]}% <br/>
       ${dataDimName.flops.display}: ${axisFormatterConstructor(3)(params.data[dataDimName.flops.dim])} <br/>
       ${dataDimName.params.display}: ${axisFormatterConstructor(3)(params.data[dataDimName.params.dim])} <br/>
       ${dataDimName.activations.display}: ${axisFormatterConstructor(3)(params.data[dataDimName.activations.dim])}`
    }
  }
}

// echarts options for scatter plot
const scatterOptions = {
  xAxis: {
    name: 'Flops',
    type: 'log',
    logBase: 4,
  },
  yAxis: {
    name: 'Acc@1',
    type: 'value',
    interval: 5
  },
  toolbox: {
    left: 'center',
    feature: {
      dataZoom: {
        show: true,
        xAxisIndex: 0,
        yAxisIndex: 0
      }
    }
  },
  ...commonOptions
}

// echarts options for parallel axis plot
const parallelOptions = {
  parallel: {
    parallelAxisDefault: {
      nameLocation: 'end',
      nameGap: 20
    }
  },
  parallelAxis: [
    { dim: dataDimName.acc1.dim, name: dataDimName.acc1.display, type: 'value', interval: 5 },
    { dim: dataDimName.flops.dim, name: dataDimName.flops.display, type: 'log', logBase: 4 },
    { dim: dataDimName.params.dim, name: dataDimName.params.display, type: 'log', logBase: 4 },
    { dim: dataDimName.activations.dim, name: dataDimName.activations.display, type: 'log', logBase: 4 }
  ],
  ...commonOptions
}

export default {
    data () {
      return {
        scatterSeries: [],
        parallelSeries: [],
        scatter_flops_log: true, // true for log space, false for linear space
        chart_type: 'scatter', // scatter or parallel
        axisDataOptions: ['flops', 'params', 'activations'],
        axisDataSelect: 'flops'
      }
    },
    computed: {
      option: function () {
        let options = null
        if (this.chart_type === 'scatter' && this.scatterSeries.length !== 0) {
          // log option
          options = { ...scatterOptions }
          options.series = this.scatterSeries
          if (this.scatter_flops_log) {
            options.xAxis.type = 'log'
          } else {
            options.xAxis.type = 'value'
          }
          // axis data
          options.xAxis.name = dataDimName[this.axisDataSelect].display
          options.series.forEach((seriesData) => {
            seriesData.encode.x = this.axisDataSelect
          })
          // axis min/max
          options.xAxis = { ...options.xAxis, ...this.calculateBoundaryWithGap(options.series, dataDimName[this.axisDataSelect].dim, options.xAxis.type) }
          options.yAxis = { ...options.yAxis, ...this.calculateBoundaryWithGap(options.series, dataDimName['acc1'].dim, options.yAxis.type) }
          // axis interval
          if (this.scatter_flops_log) {
            // FIXME: no use, why?
            options.xAxis.splitNumber = 5
          } else {
            options.xAxis.interval = (options.xAxis.max - options.xAxis.min) / 8
          }
          // axis formatter
          options.xAxis.axisLabel = { formatter: axisFormatterConstructor(1) }
        } else if (this.chart_type === 'parallel' && this.parallelSeries.length !== 0) {
          // log space or linear space
          options = { ...parallelOptions }
          options.series = this.parallelSeries
          const logAxis = ['Flops', 'Activations', 'Params'].map((name) => parallelOptions.parallelAxis.findIndex(d => d.name === name))
          if (this.scatter_flops_log) {
            for (const idx of logAxis) {
              options.parallelAxis[idx].type = 'log'
            }
          } else {
            for (const idx of logAxis) {
              options.parallelAxis[idx].type = 'value'
            }
          }
          // dynamic axis min/max and axis formatter
          options.parallelAxis = options.parallelAxis.map((axisOption) => {
            const option = { ...axisOption, ...this.calculateBoundaryWithGap(options.series, axisOption.dim, axisOption.type) }
            if (!option.name.startsWith('Acc')) {
              option.axisLabel = { formatter: axisFormatterConstructor(1) }
            }
            return option
          })
        }
        return options
      },
      dataDimName: function () {
        return dataDimName
      }
    },
    methods: {
      // parse raw json to series data
      parseJson (json) {
        const scatterSeries = {}
        const parallelSeries = {}
        let data = []
        const dataDimNameArray = Object.values(dataDimName).sort((a, b) => a.dim - b.dim)
        json.models.forEach(model => {
          data = dataDimNameArray.map(function (d) {
            if (d.name === 'url') {
              return `public/${model.uid}.html`
            } else {
              return model[d.name]
            }
          })

          scatterSeries[model.series] = scatterSeries[model.series] || {
            type: 'scatter',
            name: model.series,
            symbolSize: 10,
            itemStyle: {},
            cursor: 'pointer',
            dimensions: dataDimNameArray.map(d => d.name),
            encode: {
              x: 'flops',
              y: 'acc1'
            },
            data: []
          }
          scatterSeries[model.series].data.push(data)

          parallelSeries[model.series] = parallelSeries[model.series] || {
            type: 'parallel',
            name: model.series,
            cursor: 'pointer',
            dimensions: dataDimNameArray.map(d => d.name),
            data: [],
            opacity: 0.6,
            emphasis: {
              lineStyle: {
                width: 4,
                opacity: 1
              }
            }
          }
          parallelSeries[model.series].data.push(data)
        })

        return {
          scatterSeries: Object.values(scatterSeries),
          parallelSeries: Object.values(parallelSeries)
        }
      },

      redirectUrl (params) {
        if (params.data[params.data.length - 1]) {
          window.open(params.data[params.data.length - 1])
        }
      },

      /**
      * Dynamically calculate the axis boundary (min/max)
      *
      * @param {object} series Series data
      * @param {number} dim The dimension of axis in series
      * @param {string} type Axis type(value or log)
      * @param {number} gap The gap of boundary(0.2 means 20%)
      * @return {object} {min, max}
      */
      calculateBoundaryWithGap (series, dim, type = 'value', gap = 0.2) {
        const data = series.map(d => d.data.map(d => d[dim])).flat()
        let min = Math.min(...data)
        const oriMin = min
        let max = Math.max(...data)
        const oriMax = max
        const bound = max - min
        min = Math.floor(min - bound * gap)
        max = Math.ceil(max + bound * gap)
        if (type === 'log') {
          // min
          let n = Math.pow(2, Math.ceil(Math.log2(oriMin)))
          min = n / 2
          // max
          n = Math.pow(2, Math.floor(Math.log2(oriMax)))
          max = n * 2
        } else {
          min = Math.max(0, min)
        }
        return { min, max }
      }
    },
    beforeUpdate () {
      this.$refs.chart.clear()
    },
    mounted () {
      fetch(`${this.$store.getters.APIBase}/json/data.json`)
        .then(response => response.json())
        .then(data => {
          const series = this.parseJson(data)
          this.scatterSeries = series.scatterSeries
          this.parallelSeries = series.parallelSeries
        })
    }
}
</script>

<style>
  #chartContainer {
    height: 500px;
    width: 100%;
  }

  #vue-container {
      margin: 0 0 10px 0;
  }

  #chartOptions {
      display: flex;
      justify-content: flex-start;
      align-items: center;
  }
</style>
