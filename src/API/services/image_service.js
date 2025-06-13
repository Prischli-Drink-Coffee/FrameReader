import { Instance } from "../instance";
import { imagesUrl } from "../apiConsts";
import { getToken } from "../helper/userCookie";

//responseType: "blob",
export default class ImageService {
  static getImageBlob(image) {
    return Instance.get(`${imagesUrl}/${image}`, {
      responseType: "blob",
      headers: { Authorization: getToken() },
    });
  }

  static getImage(image) {
    return Instance.get(`${imagesUrl}/${image}`, {
      headers: { Authorization: getToken() },
    });
  }
}
