import axios from "axios";
import { baseUrl } from "./apiConsts";

export const Instance = axios.create({
  // baseURL: process.env.REACT_APP_BASE_URL,
  baseURL: baseUrl,
});
