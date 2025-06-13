export const convertDateToYesterday = (dateString) => {
  const date = new Date(dateString);
  const day = date.getUTCDate();
  const month = date.getUTCMonth() + 1;
  const year = date.getUTCFullYear();
  const formattedDate = `${(day < 10 ? "0" : "") + day}.${
    (month < 10 ? "0" : "") + month
  }.${year}`;
  return formattedDate;
};

const toPrettyDate = (date) => `${(date < 10 ? "0" : "") + date}`;

export const convertDateString = (dateString) => {
  const date = new Date(dateString);
  const day = date.getUTCDate();
  const month = date.getUTCMonth() + 1;
  const year = date.getUTCFullYear();
  const hours = date.getUTCHours();
  const minutes = date.getUTCMinutes();
  const seconds = date.getUTCSeconds();
  const formattedDate = `${toPrettyDate(day)}.${toPrettyDate(
    month,
  )}.${year} ${toPrettyDate(hours)}:${toPrettyDate(minutes)}:${toPrettyDate(
    seconds,
  )}`;
  return formattedDate;
};
