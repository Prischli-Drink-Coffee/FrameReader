import { usersUrl } from "../apiConsts";
import { Instance } from "../instance";
import { getToken } from "../helper/userCookie";

export default class UserService {
  static signIn(email, password) {
    return Instance.post(`${usersUrl.users}/${usersUrl.signIn}`, {
      email,
      password,
    });
  }
  static signUp(userSignUp) {
    return Instance.post(`${usersUrl.users}/${usersUrl.signUp}`, userSignUp, {
      headers: { Authorization: getToken() },
    });
  }
  static me(token) {
    return Instance.get(`${usersUrl.users}/${usersUrl.me}`, {
      headers: { Authorization: getToken() || `Bearer ${token}` },
    });
  }
  static getAll(page, size) {
    const params = {
      page,
      size,
    };
    return Instance.get(`${usersUrl.users}`, {
      params,
      headers: { Authorization: getToken() },
    });
  }
  static update(userId, user) {
    return Instance.put(`${usersUrl.users}/${userId}`, user, {
      headers: { Authorization: getToken() },
    });
  }
}
