// src/router.js or src/router/index.js

import {createRouter, createWebHistory} from 'vue-router'
import ImageSearch from '@/components/ImageSearch.vue'

const routes = [
    {
        path: '/',
        component: ImageSearch
    }
]

const router = createRouter({
    history: createWebHistory(),
    routes
})

export default router
