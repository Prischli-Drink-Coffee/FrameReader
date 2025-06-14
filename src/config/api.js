const API_CONFIG = {
    development: {
      baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8010/server',
      timeout: 10000,
    },
    production: {
      baseURL: process.env.REACT_APP_API_URL || '/server',
      timeout: 15000,
    }
  };
  
  const currentConfig = API_CONFIG[process.env.NODE_ENV] || API_CONFIG.development;
  
  export default currentConfig;