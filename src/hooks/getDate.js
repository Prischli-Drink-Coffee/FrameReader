export const getDate = () => {
  let currentDate = new Date();
  currentDate.setDate(currentDate.getDate() - 1);
  return currentDate;
};
