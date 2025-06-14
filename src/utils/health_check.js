import { Instance } from '../api/instance';

export const checkServerHealth = async () => {
  try {
    const response = await Instance.get('/health', { timeout: 5000 });
    return { status: 'healthy', data: response.data };
  } catch (error) {
    if (error.code === 'ERR_NETWORK') {
      return { 
        status: 'unreachable', 
        message: 'Сервер недоступен. Проверьте что FastAPI запущен на правильном порту.' 
      };
    }
    return { status: 'error', message: error.message };
  }
};