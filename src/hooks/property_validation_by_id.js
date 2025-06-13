export default function usePropertyValidationById(
  mapPropertiesValidation,
  setMapPropertiesValidation,
) {
  // Функция для смены статуса валидации у свойства в массиве listProperties, notation в данном случае значение false или true
  const helperSetListPropertiesValidation = (propertyId, notation) => {
    mapPropertiesValidation.set(propertyId, notation);
    setMapPropertiesValidation(mapPropertiesValidation);
  };

  // Проверка на заполненность для listProperties
  const emptyValidation = (value, propertyId) => {
    if (value === "") {
      helperSetListPropertiesValidation(propertyId, false);
    } else {
      helperSetListPropertiesValidation(propertyId, true);
    }
  };

  const doubleСhangeability = (value, propertyId) => {
    const validated = value.match(/^(\d*\.{0,1}\d*$)/);
    if (validated && value[0] !== "0") {
      emptyValidation(value, propertyId);
      return true;
    }
    return false;
  };

  const integerСhangeability = (value, propertyId) => {
    const validated = value.match(/^(\d*$)/);
    if (validated && value[0] !== "0") {
      emptyValidation(value, propertyId);
      return true;
    }
    return false;
  };

  const booleanСhangeability = (value, propertyId) => {
    if (typeof value === "boolean") {
      emptyValidation(value, propertyId);
      return true;
    }
    return false;
  };

  const dateСhangeability = (value, propertyId) => {
    const validated = value.match(/^\d{4}-\d{2}-\d{2}$/);
    if (validated) {
      emptyValidation(value, propertyId);
      return true;
    } else if (value === "") {
      helperSetListPropertiesValidation(propertyId, false);
      return true;
    }
    return false;
  };

  // Здесь задаётся то каким образом доллжно проверяться свойство с определённым типом
  const propertycСhangeability = (value, propertyId, type) => {
    switch (type) {
      case "STRING":
        emptyValidation(value, propertyId);
        return true;
      case "DOUBLE":
        return doubleСhangeability(value, propertyId);
      case "INTEGER":
        return integerСhangeability(value, propertyId);
      case "BOOLEAN":
        return booleanСhangeability(value, propertyId);
      case "DATE":
        return dateСhangeability(value, propertyId);
      default:
        return false;
    }
  };

  const changeMapPropertiesValidation = (properties) => {
    const newMapPropertiesValidation = new Map();
    properties.forEach((property) => {
      if (mapPropertiesValidation.has(property.id)) {
        newMapPropertiesValidation.set(
          property.id,
          mapPropertiesValidation.get(property.id),
        );
      } else {
        newMapPropertiesValidation.set(property.id, false);
      }
    });
    setMapPropertiesValidation(newMapPropertiesValidation);
  };

  return [propertycСhangeability, changeMapPropertiesValidation];
}
