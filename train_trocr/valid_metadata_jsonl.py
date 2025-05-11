#!/usr/bin/env python3
"""
Скрипт для проверки и исправления JSONL файлов датасета Donut.
Поддерживает обработку файлов с различными кодировками и удаление проблемных записей.
"""
import os
import glob
import json
import argparse
import chardet
from typing import List, Dict, Tuple, Optional, Any
import logging


class JSONLValidator:
    """
    Класс для валидации и исправления JSONL файлов.
    """
    
    def __init__(self, fix: bool = False, verbose: bool = False, remove_broken: bool = True):
        """
        Инициализирует валидатор JSONL файлов.
        
        Args:
            fix: Если True, исправлять найденные ошибки
            verbose: Если True, выводить подробную информацию о процессе
            remove_broken: Если True, удалять невосстановимые записи вместо создания новых
        """
        self.fix = fix
        self.verbose = verbose
        self.remove_broken = remove_broken
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """
        Настраивает логгер для вывода информации.
        
        Returns:
            Настроенный логгер
        """
        logger = logging.getLogger("jsonl_validator")
        logger.setLevel(logging.INFO)
        
        # Очистка обработчиков, если они уже существуют
        if logger.handlers:
            logger.handlers.clear()
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Вывод в консоль
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def detect_encoding(self, file_path: str) -> str:
        """
        Определяет кодировку файла.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Кодировка файла
        """
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Читаем первые 10000 байт для определения кодировки
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            
            if self.verbose:
                self.logger.info(f"Обнаружена кодировка {encoding} с уверенностью {confidence:.2f} для файла {file_path}")
            
            # Если кодировка не определена или определена с низкой уверенностью,
            # пробуем определить по большему фрагменту
            if encoding is None or confidence < 0.7:
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    result = chardet.detect(raw_data)
                    encoding = result['encoding']
                    confidence = result['confidence']
                    
                    if self.verbose:
                        self.logger.info(f"Повторное определение: кодировка {encoding} с уверенностью {confidence:.2f}")
            
            # Если всё еще не удалось определить кодировку, используем Latin-1 как безопасную опцию
            if encoding is None:
                self.logger.warning(f"Не удалось определить кодировку файла {file_path}, используем Latin-1")
                return 'latin-1'
            
            return encoding
    
    def read_file_with_encoding(self, file_path: str) -> List[str]:
        """
        Читает файл с автоматическим определением кодировки.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Список строк файла
        """
        encoding = self.detect_encoding(file_path)
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            return lines
        except UnicodeDecodeError as e:
            self.logger.error(f"Ошибка декодирования с использованием {encoding}: {e}")
            self.logger.info("Пробуем использовать latin-1 как запасной вариант")
            
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    lines = f.readlines()
                return lines
            except Exception as e:
                self.logger.error(f"Не удалось прочитать файл {file_path}: {e}")
                return []
    
    def validate_json_line(self, line: str, line_number: int, file_path: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Проверяет и пытается исправить строку JSON.
        
        Args:
            line: Строка JSON
            line_number: Номер строки
            file_path: Путь к файлу (для логов)
            
        Returns:
            Кортеж (is_valid, fixed_json)
        """
        line = line.strip()
        if not line:
            if self.verbose:
                self.logger.warning(f"Пустая строка в файле {file_path} на строке {line_number}")
            return False, None
        
        try:
            entry = json.loads(line)
            
            # Проверка наличия необходимых полей
            missing_fields = []
            if "file_name" not in entry:
                missing_fields.append("file_name")
            
            if "ground_truth" not in entry:
                missing_fields.append("ground_truth")
            else:
                # Проверка, что ground_truth содержит валидный JSON
                try:
                    gt = json.loads(entry["ground_truth"])
                    if "gt_parse" not in gt:
                        if self.verbose:
                            self.logger.warning(f"Отсутствует поле 'gt_parse' в ground_truth в файле {file_path} на строке {line_number}")
                        
                        if self.fix:
                            # Добавляем пустой gt_parse, если его нет
                            gt["gt_parse"] = {}
                            entry["ground_truth"] = json.dumps(gt)
                            return True, entry
                except json.JSONDecodeError:
                    if self.verbose:
                        self.logger.warning(f"Некорректный JSON в поле 'ground_truth' в файле {file_path} на строке {line_number}")
                    
                    if self.fix:
                        # Пытаемся исправить ground_truth
                        try:
                            # Если это строка в кавычках, пробуем распарсить её
                            if isinstance(entry["ground_truth"], str):
                                # Заменяем одинарные кавычки на двойные
                                fixed_gt = entry["ground_truth"].replace("'", "\"")
                                # Пробуем парсить
                                try:
                                    gt = json.loads(fixed_gt)
                                    if "gt_parse" not in gt:
                                        gt["gt_parse"] = {}
                                    entry["ground_truth"] = json.dumps(gt)
                                    return True, entry
                                except json.JSONDecodeError:
                                    if self.remove_broken:
                                        self.logger.warning(f"Удаляем запись с невосстановимым JSON в поле 'ground_truth' в файле {file_path} на строке {line_number}")
                                        return False, None
                                    else:
                                        # Если не удаётся исправить, создаём новый объект
                                        entry["ground_truth"] = json.dumps({"gt_parse": {}})
                                        return True, entry
                        except Exception:
                            if self.remove_broken:
                                self.logger.warning(f"Удаляем запись с невосстановимым JSON в поле 'ground_truth' в файле {file_path} на строке {line_number}")
                                return False, None
                            else:
                                # Если не удалось исправить, создаём новый объект
                                entry["ground_truth"] = json.dumps({"gt_parse": {}})
                                return True, entry
            
            if missing_fields:
                if self.verbose:
                    self.logger.warning(f"Отсутствуют поля {', '.join(missing_fields)} в файле {file_path} на строке {line_number}")
                
                if self.fix:
                    if self.remove_broken and len(missing_fields) > 1:  # Если отсутствуют оба обязательных поля
                        self.logger.warning(f"Удаляем запись с отсутствующими обязательными полями в файле {file_path} на строке {line_number}")
                        return False, None
                    
                    # Добавляем отсутствующие поля
                    if "file_name" not in entry:
                        entry["file_name"] = f"unknown_{line_number}.png"
                    
                    if "ground_truth" not in entry:
                        entry["ground_truth"] = json.dumps({"gt_parse": {}})
                    
                    return True, entry
                
                return False, None
            
            return True, entry
            
        except json.JSONDecodeError:
            self.logger.warning(f"Некорректный JSON в файле {file_path} на строке {line_number}")
            
            if not self.fix:
                return False, None
            
            # Попытка исправить JSON
            try:
                # Попытка исправления типичных ошибок
                fixed_line = line
                
                # Проверка на незакрытые скобки
                if fixed_line.count('{') > fixed_line.count('}'):
                    fixed_line += '}' * (fixed_line.count('{') - fixed_line.count('}'))
                
                # Проверка на незакрытые кавычки (исправляем только если нечетное число)
                if fixed_line.count('"') % 2 != 0:
                    fixed_line += '"'
                
                # Попытка парсинга исправленной строки
                try:
                    entry = json.loads(fixed_line)
                    self.logger.info(f"Успешно исправлен JSON в файле {file_path} на строке {line_number}")
                    
                    # Проверка и исправление отсутствующих полей
                    has_required_fields = True
                    if "file_name" not in entry:
                        has_required_fields = False
                    
                    if "ground_truth" not in entry:
                        has_required_fields = False
                    
                    if not has_required_fields and self.remove_broken:
                        self.logger.warning(f"Удаляем запись с отсутствующими обязательными полями в файле {file_path} на строке {line_number}")
                        return False, None
                    
                    # Исправление полей, если не удаляем запись
                    if "file_name" not in entry:
                        entry["file_name"] = f"unknown_{line_number}.png"
                    
                    if "ground_truth" not in entry:
                        entry["ground_truth"] = json.dumps({"gt_parse": {}})
                    else:
                        try:
                            gt = json.loads(entry["ground_truth"])
                            if "gt_parse" not in gt:
                                gt["gt_parse"] = {}
                                entry["ground_truth"] = json.dumps(gt)
                        except json.JSONDecodeError:
                            if self.remove_broken:
                                self.logger.warning(f"Удаляем запись с невосстановимым JSON в поле 'ground_truth' в файле {file_path} на строке {line_number}")
                                return False, None
                            else:
                                entry["ground_truth"] = json.dumps({"gt_parse": {}})
                    
                    return True, entry
                except json.JSONDecodeError:
                    if self.remove_broken:
                        self.logger.warning(f"Удаляем запись с невосстановимым JSON в файле {file_path} на строке {line_number}")
                        return False, None
                    else:
                        # Если не удается исправить, создаем новую запись
                        self.logger.warning(f"Не удалось исправить JSON в файле {file_path} на строке {line_number}, создаем новую запись")
                        return True, {"file_name": f"unknown_{line_number}.png", "ground_truth": json.dumps({"gt_parse": {}})}
            
            except Exception as e:
                self.logger.error(f"Ошибка при попытке исправить JSON в файле {file_path} на строке {line_number}: {e}")
                if self.remove_broken:
                    self.logger.warning(f"Удаляем запись с невосстановимым JSON в файле {file_path} на строке {line_number}")
                    return False, None
                else:
                    return False, None
    
    def validate_jsonl_file(self, file_path: str) -> bool:
        """
        Проверяет и исправляет JSONL файл.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            True, если файл был успешно проверен и исправлен (если fix=True)
        """
        self.logger.info(f"Проверка файла: {file_path}")
        
        try:
            lines = self.read_file_with_encoding(file_path)
            
            if not lines:
                self.logger.error(f"Не удалось прочитать файл {file_path} или файл пуст")
                return False
            
            valid_entries = []
            invalid_count = 0
            empty_count = 0
            removed_count = 0
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    empty_count += 1
                    continue
                
                is_valid, fixed_entry = self.validate_json_line(line, i + 1, file_path)
                
                if is_valid and fixed_entry:
                    valid_entries.append(fixed_entry)
                else:
                    invalid_count += 1
                    if self.remove_broken:
                        removed_count += 1
            
            if invalid_count > 0 or empty_count > 0:
                if self.remove_broken and removed_count > 0:
                    self.logger.info(f"Найдено {invalid_count} некорректных строк, удалено {removed_count} невосстановимых записей и {empty_count} пустых строк в файле {file_path}")
                else:
                    self.logger.info(f"Найдено {invalid_count} некорректных строк и {empty_count} пустых строк в файле {file_path}")
                
                if self.fix and valid_entries:
                    # Создаем бэкап файла перед перезаписью
                    backup_path = file_path + ".bak"
                    try:
                        import shutil
                        shutil.copy2(file_path, backup_path)
                        self.logger.info(f"Создан бэкап файла: {backup_path}")
                    except Exception as e:
                        self.logger.error(f"Не удалось создать бэкап файла: {e}")
                    
                    # Записываем исправленные данные
                    try:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            for entry in valid_entries:
                                f.write(json.dumps(entry) + '\n')
                        
                        self.logger.info(f"Файл исправлен: записано {len(valid_entries)} валидных строк")
                        return True
                    except Exception as e:
                        self.logger.error(f"Ошибка при записи исправленного файла: {e}")
                        return False
            else:
                self.logger.info(f"Файл корректен: {len(valid_entries)} строк")
                return True
        
        except Exception as e:
            self.logger.error(f"Ошибка при обработке файла {file_path}: {e}")
            return False
    
    def validate_jsonl_files(self, directory_path: str) -> None:
        """
        Проверяет все JSONL файлы в указанной директории.
        
        Args:
            directory_path: Путь к директории с JSONL файлами
        """
        jsonl_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".jsonl"):
                    jsonl_files.append(os.path.join(root, file))
        
        self.logger.info(f"Найдено {len(jsonl_files)} JSONL файлов")
        
        success_count = 0
        for file_path in jsonl_files:
            if self.validate_jsonl_file(file_path):
                success_count += 1
        
        self.logger.info(f"Обработано {len(jsonl_files)} файлов, успешно: {success_count}")


def convert_file_encoding(file_path: str, target_encoding: str = 'utf-8') -> bool:
    """
    Конвертирует файл в указанную кодировку.
    
    Args:
        file_path: Путь к файлу
        target_encoding: Целевая кодировка
        
    Returns:
        True, если конвертация выполнена успешно
    """
    try:
        # Определяем текущую кодировку
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            source_encoding = result['encoding']
            
            if source_encoding == target_encoding:
                print(f"Файл {file_path} уже в кодировке {target_encoding}")
                return True
            
            print(f"Конвертация файла {file_path} из {source_encoding} в {target_encoding}")
            
            # Декодируем содержимое файла и перекодируем в целевую кодировку
            content = raw_data.decode(source_encoding, errors='replace')
            
            # Создаем бэкап файла
            backup_path = file_path + f".{source_encoding}.bak"
            import shutil
            shutil.copy2(file_path, backup_path)
            print(f"Создан бэкап файла: {backup_path}")
            
            # Записываем в новой кодировке
            with open(file_path, 'w', encoding=target_encoding) as f:
                f.write(content)
            
            print(f"Файл успешно конвертирован в {target_encoding}")
            return True
    
    except Exception as e:
        print(f"Ошибка при конвертации файла {file_path}: {e}")
        return False


def main():
    """
    Основная функция скрипта.
    """
    parser = argparse.ArgumentParser(description="Проверка и исправление JSONL файлов датасета Donut")
    parser.add_argument("dir", help="Путь к директории с JSONL файлами")
    parser.add_argument("--fix", action="store_true", help="Исправлять найденные ошибки")
    parser.add_argument("--verbose", action="store_true", help="Подробный вывод")
    parser.add_argument("--convert", action="store_true", help="Конвертировать файлы в UTF-8")
    parser.add_argument("--keep-broken", action="store_true", help="Сохранять невосстановимые записи, создавая для них новые")
    args = parser.parse_args()
    
    if args.convert:
        # Конвертируем все JSONL файлы в UTF-8
        jsonl_files = []
        for root, _, files in os.walk(args.dir):
            for file in files:
                if file.endswith(".jsonl"):
                    jsonl_files.append(os.path.join(root, file))
        
        print(f"Найдено {len(jsonl_files)} JSONL файлов для конвертации")
        
        for file_path in jsonl_files:
            convert_file_encoding(file_path, 'utf-8')
    
    # Запускаем валидацию
    validator = JSONLValidator(fix=args.fix, verbose=args.verbose, remove_broken=not args.keep_broken)
    validator.validate_jsonl_files(args.dir)


if __name__ == "__main__":
    main()