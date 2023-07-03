<template>
  <div>
    <input type="file" @change="handleFileUpload"/>
    <button @click="search">Search</button>
    <div v-if="searchResult">
      <p>Searched Image:</p>
      <div class="search-image-container" :style="bboxStyle(searchResult.bbox)">
        <img :src="searchResult.search_img" alt="Searched Image" @load="imageLoaded"/>
      </div>
      <div>
        Bounding Box: {{ searchResult.bbox.join(',') }}
        Label: {{ searchResult.label }}
        Bounding Box Score: {{ searchResult.bbox_score }}
      </div>
      <p>Similar Images:</p>
      <div v-for="result in searchResult.results" :key="result.image_key">
        <div class="result-image-container" :style="bboxStyle(result.box.split(',').map(Number))">
          <img :src="result.image_key" alt="Result Image" @load="imageLoaded"/>
        </div>
        <div>
          Box: {{ result.box }}
          Label: {{ result.label }}
          Score: {{ result.score }}
        </div>
      </div>
    </div>
  </div>
</template>


<script>
import {ref} from 'vue';
import axios from 'axios';
import {createThumbnail} from '@/utils/image.js';

export default {
  setup() {
    const searchResult = ref(null);
    const imageFile = ref(null);

    const handleFileUpload = (event) => {
      imageFile.value = event.target.files[0];
    };

    const search = async () => {
      if (!imageFile.value) {
        return;
      }
      try {
        const reader = new FileReader();
        reader.onload = async (e) => {
          let imageData = e.target.result;
          imageData = await createThumbnail(imageData, 450);

          const byteCharacters = atob(imageData.split(',')[1]);
          const byteNumbers = new Array(byteCharacters.length);
          for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
          }
          const byteArray = new Uint8Array(byteNumbers);
          const imageBlob = new Blob([byteArray], {type: 'image/jpeg'});
          const formData = new FormData();
          formData.append('file', imageBlob, 'file.jpg');

          const response = await axios.post(`http://localhost:8090/search`, formData, {
            headers: {
              'Content-Type': 'multipart/form-data'
            }
          });
          searchResult.value = response.data.data;
        };
        reader.readAsDataURL(imageFile.value);
      } catch (error) {
        console.error(error);
      }
    };

    const imageLoaded = () => {
      // force a re-render when the image finishes loading
      searchResult.value = {...searchResult.value};
    };

    const bboxStyle = (bbox) => {
      return {
        '--bbox-left': bbox[0] + 'px',
        '--bbox-top': bbox[1] + 'px',
        '--bbox-width': (bbox[2] - bbox[0]) + 'px',
        '--bbox-height': (bbox[3] - bbox[1]) + 'px',
      };
    };

    return {
      searchResult,
      search,
      handleFileUpload,
      imageLoaded,
      bboxStyle
    };
  }
};
</script>

<style scoped>
.search-image-container::before,
.result-image-container::before {
  content: "";
  position: absolute;
  left: var(--bbox-left);
  top: var(--bbox-top);
  width: var(--bbox-width);
  height: var(--bbox-height);
  border: 2px solid red;
  pointer-events: none;
}

.search-image-container,
.result-image-container {
  position: relative;
  display: inline-block;
}


</style>
