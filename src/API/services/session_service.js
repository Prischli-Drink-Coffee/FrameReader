import { Instance } from "../instance";

class SessionServiceError extends Error {
  constructor(message, code = null, originalError = null) {
    super(message);
    this.name = 'SessionServiceError';
    this.code = code;
    this.originalError = originalError;
  }
}

class SessionService {
  static async createOrGetSession() {
    try {
      const response = await Instance.post("/auth/session/create");
      
      if (!response?.data) {
        throw new SessionServiceError("Invalid response from server", "INVALID_RESPONSE");
      }
      
      return response.data;
    } catch (error) {
      if (error instanceof SessionServiceError) {
        throw error;
      }
      
      if (error.code === 'ERR_NETWORK') {
        throw new SessionServiceError(
          'Не удается подключиться к серверу. Проверьте, что API сервер запущен.',
          'NETWORK_ERROR',
          error
        );
      }
      
      if (error.response?.status) {
        throw new SessionServiceError(
          `Server error: ${error.response.status}`,
          'SERVER_ERROR',
          error
        );
      }
      
      throw new SessionServiceError(
        'Unexpected error creating session',
        'UNKNOWN_ERROR',
        error
      );
    }
  }

  static async getCurrentSessionInfo() {
    try {
      const response = await Instance.get("/auth/session/info");
      
      if (!response?.data) {
        throw new SessionServiceError("Invalid response from server", "INVALID_RESPONSE");
      }
      
      return response.data;
    } catch (error) {
      if (error instanceof SessionServiceError) {
        throw error;
      }
      
      if (error.response?.status === 401) {
        throw new SessionServiceError('Сессия не найдена или истекла', 'SESSION_EXPIRED', error);
      }
      
      throw new SessionServiceError(
        'Error getting session info',
        'GET_SESSION_INFO_FAILED',
        error
      );
    }
  }

  static async validateSession() {
    try {
      const response = await Instance.get("/auth/session/validate");
      
      if (!response?.data) {
        return { valid: false };
      }
      
      return response.data;
    } catch (error) {
      if (error.response?.status === 401) {
        return { valid: false };
      }
      
      console.warn("Session validation error:", error);
      return { valid: false };
    }
  }

  static async logoutSession() {
    try {
      const response = await Instance.post("/auth/session/logout");
      return response?.data || { message: "Logged out" };
    } catch (error) {
      console.warn("Logout error (might be expected):", error);
      return { message: "Logged out" };
    }
  }
}

export default SessionService;
export { SessionServiceError };