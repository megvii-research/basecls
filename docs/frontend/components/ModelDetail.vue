<template>
    <el-descriptions title="" :column="2" border>
      <el-descriptions-item v-for="(value, key) in tabledata" :key="key" :label="key">
        {{ value }}
      </el-descriptions-item>
      <el-descriptions-item v-if="uid" label="download">
        <el-link :href="url" type="primary" :underline="false">{{ uid }}.pkl</el-link>
      </el-descriptions-item>
   </el-descriptions>
</template>

<script>
export default {
          data () {
        return {
          uid: null,
          tabledata: {}
        }
      },

      computed: {
        url: function () {
          return `${this.$store.getters.OSSPath}/${this.uid.split("/")[1]}/${this.uid}.pkl`
        }
      },

      mounted () {
        const downloadUid = document.getElementById('modelInfoContainer').dataset.uid
        fetch(`${this.$store.getters.APIBase}/json/${downloadUid}.json`)
          .then(response => {
            if (response.ok) {
              return response.json()
            } else if (response.status === 404) {
              this.tabledata = { info: 'model not exist' }
              throw new Error('model not exist')
            } else {
              this.tabledata = { info: 'internal error' }
              throw new Error('internal error')
            }
          })
          .then((data) => {
            this.uid = data.uid

            this.tabledata = { ...data }
            delete this.tabledata.uid
          })
          .catch((error) => {
            console.error(error)
          })
      }
}
</script>
