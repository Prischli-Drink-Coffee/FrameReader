import Cookies from "js-cookie";

export const getToken = () => {
  const token = Cookies.get("token");
  if (token === undefined) {
    return token;
  }
  return `Bearer ${token}`;
};

export const setUser = (user) => {
  Cookies.set("token", user.token, { expires: 365 ** 2 });
  Cookies.set("userEmail", user.userEmail, { expires: 365 ** 2 });
};

export const deleteUser = () => {
  Cookies.remove("token");
  Cookies.remove("userEmail");
};
