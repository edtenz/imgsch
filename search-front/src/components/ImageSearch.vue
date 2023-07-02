<template>
  <div>
    <input type="text" v-model="searchQuery"/>
    <button @click="search">Search</button>
    <ul>
      <li v-for="result in searchResults" :key="result.id">{{ result.name }}</li>
    </ul>
  </div>
</template>

<script>
import {ref} from 'vue';
import axios from 'axios';

export default {
  name: 'ImageSearch',
  setup() {
    const searchQuery = ref('');
    const searchResults = ref([]);

    const search = async () => {
      try {
        const response = await axios.get(`http://localhost:8090/test?q=${searchQuery.value}`);
        searchResults.value = response.data;
      } catch (error) {
        console.error(error);
      }
    };

    return {
      searchQuery,
      searchResults,
      search
    };
  }
};
</script>

<style>
/* Add any necessary CSS */
</style>
