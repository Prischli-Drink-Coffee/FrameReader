import { baseUrl } from "../apiConsts";


/**
 * API для вызова эндпоинта /inference/
 *
 * @param {string} videoUrl - URL видео для инференса.
 * @returns {Promise<Object>} Ответ от сервера.
 * @throws {Error} Если запрос завершился ошибкой.
 */
export async function sendInference(videoUrl) {
  const API_ENDPOINT = `${baseUrl}/inference/?url=${encodeURIComponent(videoUrl)}`; // Кодируем URL для безопасности

  try {
    const response = await fetch(API_ENDPOINT, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(`Error: ${response.statusText}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error in inference request:", error);
    throw error;
  }
}