import { useState } from "react";

export const useFetching = (callback) => {
  const [error, setError] = useState("");

  const fetching = async () => {
    try {
      await callback();
      setError();
    } catch (e) {
      setError(e.message);
    }
  };
  return [fetching, error];
};
