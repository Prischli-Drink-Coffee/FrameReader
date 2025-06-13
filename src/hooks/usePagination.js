import { useMemo } from "react";

const paginationPages = 5;
export const usePagination = (totalPages, currentPage) =>
  useMemo(() => {
    const pageArray = [];
    if (totalPages > paginationPages) {
      if (currentPage > 3) {
        if (totalPages - 2 < currentPage) {
          for (let i = totalPages - paginationPages + 1; i <= totalPages; i++) {
            pageArray.push(i);
            if (i === totalPages) break;
          }
        } else {
          for (let i = currentPage - 2; i <= currentPage + 2; i++) {
            pageArray.push(i);
            if (i === totalPages) break;
          }
        }
      } else {
        for (let i = 1; i <= paginationPages; i++) {
          pageArray.push(i);
          if (i === totalPages) break;
        }
      }
    } else {
      for (let i = 1; i <= totalPages; i++) {
        pageArray.push(i);
      }
    }
    return pageArray;
  }, [totalPages, currentPage]);
