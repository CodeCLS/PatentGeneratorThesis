/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:50026/api/:path*',
      },
      {
        source: '/static/:path*',
        destination: 'http://localhost:50026/static/:path*',
      },
    ];
  },
};

module.exports = nextConfig;

