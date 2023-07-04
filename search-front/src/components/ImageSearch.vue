<template>
  <div>
    <input type="file" @change="handleFileUpload"/>
    <button @click="search">Search</button>
    <div v-if="searchResult" class="results-container">
      <!--      <div class="image-container">-->
      <!--        <canvas ref="canvas"></canvas>-->
      <!--      </div>-->
      <div class="search-image-container" :style="bboxStyle(searchResult.bbox.box)">
        <img :src="searchResult.search_img" alt="Searched Image" @load="imageLoaded"/>
      </div>
      <div>
        Bounding Box: {{ searchResult.bbox.box.join(',') }}
        Label: {{ searchResult.bbox.label }}
        Bounding Box Score: {{ searchResult.bbox.score }}
      </div>
      <p>Candidate Boxes:</p>
      <div v-for="cbox in searchResult.candidate_box" :key="cbox.box.join(',')">
        <div :style="bboxStyle(cbox.box)">
          Box: {{ cbox.box.join(',') }}
          Label: {{ cbox.label }}
          Bounding Box Score: {{ cbox.score }}
        </div>
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
  },

  mounted() {
    this.$nextTick(() => {
      this.drawImageAndBoxes();
    });
  },

  methods: {
    drawImageAndBoxes() {
      const canvas = this.$refs.canvas;
      const context = canvas.getContext('2d');
      const image = new Image();

      image.onload = () => {
        canvas.width = image.width;
        canvas.height = image.height;

        context.drawImage(image, 0, 0, image.width, image.height);

        // Draw main bounding box (bbox)
        let [x1, y1, x2, y2] = this.searchResult.bbox.box;
        context.beginPath();
        context.rect(x1, y1, x2 - x1, y2 - y1);
        context.lineWidth = 3;
        context.strokeStyle = 'red';
        context.stroke();

        // Draw candidate bounding boxes
        this.searchResult.candidate_box.forEach(cbox => {
          let [x1, y1, x2, y2] = cbox.box;
          context.beginPath();
          context.rect(x1, y1, x2 - x1, y2 - y1);
          context.lineWidth = 3;
          context.strokeStyle = 'green';
          context.stroke();
        });
      };

      image.src = this.searchResult.search_img;
    },
  },
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

.image-container {
  position: relative;
}

canvas {
  position: absolute;
  top: 0;
  left: 0;
}
</style>
