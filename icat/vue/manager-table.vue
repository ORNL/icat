<template>
  <v-data-table
    class='softhover-table'
    :height="height"
    :width="width"
    dense
    :items="items"
    :headers="headers"
    :server-items-length="total_length"
    :options="options"
    @update:options="updateOptions"
  >
    <template #body="{ items }">
      <tbody :width="width">
        <tr v-for="item in items" :key="item.id" @click="selectPoint(item.id)" @mouseover="hoverPoint(item.id)">
          <td v-html="item.text" style="padding-left: 2px; padding-right: 2px; word-break: break-word;" />
            <td style="vertical-align: top; padding-left: 2px; padding-right: 2px; color: grey;">{{ item.id }}</td>
            <td style="vertical-align: top; padding-bottom: 5px; padding-left: 2px;">
              <v-tooltip bottom open-delay=500>
                <template v-slot:activator="{ on, attrs }">
                  <v-btn x-small class="blue darken-1" @click.stop="applyAbsoluteLabelUninteresting(item.id)" v-bind="attrs" v-on="on">
                    U
                  </v-btn>
                </template>
                <span>Label this item as <span class="blue--text lighten-4"><b>uninteresting</b></span> (cold).</span>
              </v-tooltip>
              <v-tooltip bottom open-delay=500>
                <template v-slot:activator="{ on, attrs }">
                  <v-btn x-small class="orange darken-1" @click.stop="applyAbsoluteLabelInteresting(item.id)" v-bind="attrs" v-on="on">
                    I
                  </v-btn>
                </template>
                <span>Label this item as <span class="orange--text lighten-5"><b>interesting</b></span> (warm).</span>
              </v-tooltip>
              <v-tooltip bottom open-delay=500 v-if="example_btn_color != ''">
                <template v-slot:activator="{ on, attrs }">
                  <v-btn x-small :style="{ backgroundColor: example_btn_color }" @click.stop="addToExampleAnchor(item.id)" v-bind="attrs" v-on="on">
                    example
                  </v-btn>
                </template>
                <span>Create a {{ example_type_name }} anchor with this item as the target.</span>
              </v-tooltip>
              <v-tooltip bottom open-delay=500>
                <template v-slot:activator="{ on, attrs }">
                  <v-btn x-small v-if="!item.in_sample" @click.stop="addToSample(item.id)" v-bind="attrs" v-on="on">
                    sample
                  </v-btn>
                </template>
                <span>Add this item to the current sample set.</span>
              </v-tooltip>
              <v-tooltip bottom open-delay=500>
                <template v-slot:activator="{ on, attrs }">
                  <div v-html="item.labeled" v-bind="attrs" v-on="on" />
                </template>
                <span v-if="item.labeled.indexOf('orange') > -1"><span class="orange--text lighten-5"><b>interesting</b></span> (warm).</span>
                <span v-if="item.labeled.indexOf('blue') > -1"><span class="blue--text lighten-4"><b>uninteresting</b></span> (cold).</span>
              </v-tooltip>
              <v-tooltip bottom open-delay=500 v-if="item.labeled != ''">
                <template v-slot:activator="{ on, attrs }">
                  <v-btn x-small class="red darken-4" @click.stop="applyAbsoluteLabelUnlabeled(item.id)" v-bind="attrs" v-on="on">
                    unlabel
                  </v-btn>
                </template>
                <span>Remove the label from this item.</span>
              </v-tooltip>
            </td>
        </tr>
      </tbody>
    </template>
  </v-data-table>
</template>

<style id='manager-table-styles'>
.softhover-table table tbody tr:hover {
  background-color: #333333 !important;
}
.softhover-table table thead tr th {
  padding-left: 5px !important;
  padding-right: 5px !important;
}
.v-data-table__wrapper {
  overscroll-behavior: contain !important;
}
</style>
