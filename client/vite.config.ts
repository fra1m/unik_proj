import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

const inDocker = process.env.DOCKERIZED === '1';
const gatewayTarget = inDocker ? 'http://gateway:3000' : 'http://localhost:3000';

export default defineConfig({
	plugins: [react(), tailwindcss()],
	server: {
		host: '0.0.0.0',
		port: 5173,
		strictPort: true,
		proxy: {
			'/api': {
				target: gatewayTarget,
				changeOrigin: true,
				ws: true,
				rewrite: p => p.replace(/^\/api/, '/api'),
			},
		},
	},
});
