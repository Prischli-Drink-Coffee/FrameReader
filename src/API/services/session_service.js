import { Instance } from "../instance";

class SessionService {
  static async createOrGetSession() {
    try {
      const response = await Instance.post("/auth/session/create");
      return response.data;
    } catch (error) {
      if (error.code === 'ERR_NETWORK') {
        throw new Error('Не удается подключиться к серверу. Проверьте, что API сервер запущен.');
      }
      throw error;
    }
  }

  static async getCurrentSessionInfo() {
    try {
      const response = await Instance.get("/auth/session/info");
      return response.data;
    } catch (error) {
      if (error.response?.status === 401) {
        throw new Error('Сессия не найдена или истекла');
      }
      throw error;
    }
  }

  static async validateSession() {
    try {
      const response = await Instance.get("/auth/session/validate");
      return response.data;
    } catch (error) {
      if (error.response?.status === 401) {
        return { valid: false };
      }
      throw error;
    }
  }

  static async logoutSession() {
    try {
      const response = await Instance.post("/auth/session/logout");
      return response.data;
    } catch (error) {
      console.warn("Logout error (might be expected):", error);
      return { message: "Logged out" };
    }
  }
}

export default SessionService;