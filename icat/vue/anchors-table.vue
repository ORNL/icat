<template>
  <v-data-table
    :headers="headers"
    :items="items"
    hide-default-footer
    show-expand
    dense
    :loading="processing"
    :expanded.sync="expanded"
    class="dense-table striped-table"
    item-key="name"
    items-per-page="100"
  >
    <!--Note that for some reason an items-per-page="-1" breaks this, unclear on why.-->
    <template v-slot:item="{ item, expand, isExpanded }">
      <!--<tr :style="{ backgroundColor: item.color, backgroundImage: 'linear-gradient(rgb(0 0 0/40%) 0 0)' }">-->
      <!--<tr>-->
      <tr :style="{ backgroundColor: item.color, backgroundImage: 'linear-gradient(rgb(50 50 50/95%) 0 0)' }">
        <td>
          <v-btn icon x-small @click="expand(!isExpanded)" :style="{ backgroundColor: item.color }">
            <v-icon v-if="isExpanded" style="margin-left: -1px; margin-top: 1px;">mdi-chevron-down</v-icon>
            <v-icon v-if="!isExpanded" style="margin-left: -1px; margin-top: 1px;">mdi-chevron-up</v-icon>
          </v-btn>
        </td>


        <td><div><jupyter-widget :widget="item.anchor_name" /></div></td>
        <td><div>{{ item.coverage }}</div></td>
        <td><div class='blue--text darken-1'>{{ item.pct_negative }}</div></td>
        <td><div class='orange--text darken-1'>{{ item.pct_positive }}</div></td>
        <td><div><jupyter-widget :widget="item.in_viz" /></div></td>
        <td><div><jupyter-widget :widget="item.in_model" /></div></td>
        <td>
          <v-btn
              icon
              x-small
              class='delete-button'
              @click="deleteAnchor(item.name)"
              :loading="item.processing"
              >
              <v-icon>mdi-close-circle-outline</v-icon>
          </v-btn>
        </td>
      </tr>
    </template>

    <template v-slot:expanded-item="{ headers, item }">
      <tr class="v-data-table__expanded__content" :style="{ backgroundColor: item.color, backgroundImage: 'linear-gradient(rgb(0 0 0/80%) 0 0)' }">
        <td :colspan="headers.length+1">
          <jupyter-widget :widget="item.widget" />
        </td>
      </tr>
    </template>
  </v-data-table>
</template>

<style id='anchor-table-styles'>
.softhover-table table tbody tr:hover {
  background-color: #333333 !important;
}

.delete-button {
  margin: 0px;
  margin-left: 6px;
  color: var(--md-grey-500) !important;
}
.delete-button:hover {
  color: var(--md-red-500) !important;
}

.striped-table tbody tr:nth-child(even) {
  background-color: rgba(0, 0, 0, 0.35);
}
.striped-table .v-data-table__expanded__content td {
  /*background-color: #263238; */
}

.dense-table .row {
  flex-wrap: nowrap;
}
.dense-table td {
  padding: 0 4px !important;
  height: 30px !important;
  max-height: 30px !important;
  vertical-align: middle;
}
.dense-table th {
  padding: 0 4px !important;
}
.dense-table td .v-input {
  margin: 0;
  /* margin-top: 5px; */
}
.dense-table td .v-input__slot {
  margin-bottom: 0;
}
.dense-table td .v-input--selection-controls__input {
  margin-top: -5px;
}
.dense-table td .v-input--selection-controls__ripple {
  margin: 7px;
  height: 25px !important;
  width: 25px !important;
}
.dense-table td .v-icon.v-icon::after {
  transform: scale(1.2) !important;
}
.dense-table td .v-input .v-messages {
  display: none;
  height: 0;
}
.dense-table td .v-text-field__details {
  height: 2px !important;
  min-height: 2px !important;
}

div .v-progress-linear {
  left: -1px !important;
}

/* this should probably go somewhere else, these are
   vars for anchorviz. */
:host {
  --selectedAnchorColor: #FFF;
}
</style>
