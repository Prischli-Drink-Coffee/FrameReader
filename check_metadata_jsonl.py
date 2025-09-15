#!/usr/bin/env python3

import os
import glob
import json
import argparse
import chardet
from typing import List, Dict, Tuple, Optional, Any
import logging


class JSONLValidator:
    
    def __init__(self, fix: bool = False, verbose: bool = False, remove_broken: bool = True):

        self.fix = fix
        self.verbose = verbose
        self.remove_broken = remove_broken
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:

        logger = logging.getLogger("jsonl_validator")
        logger.setLevel(logging.INFO)
        
        if logger.handlers:
            logger.handlers.clear()
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def detect_encoding(self, file_path: str) -> str:

        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            
            if self.verbose:
                self.logger.info(f"Обнаружена кодировка {encoding} с уверенностью {confidence:.2f} для файла {file_path}")
            
            if encoding is None or confidence < 0.7:
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    result = chardet.detect(raw_data)
                    encoding = result['encoding']
                    confidence = result['confidence']
                    
                    if self.verbose:
                        self.logger.info(f"Повторное определение: кодировка {encoding} с уверенностью {confidence:.2f}")
            
            if encoding is None:
                self.logger.warning(f"Не удалось определить кодировку файла {file_path}, используем Latin-1")
                return 'latin-1'
            
            return encoding
    
    def read_file_with_encoding(self, file_path: str) -> List[str]:
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
        line = line.strip()
        if not line:
            if self.verbose:
                self.logger.warning(f"Пустая строка в файле {file_path} на строке {line_number}")
            return False, None
        
        try:
            entry = json.loads(line)
            
            missing_fields = []
            if "file_name" not in entry:
                missing_fields.append("file_name")
            
            if "ground_truth" not in entry:
                missing_fields.append("ground_truth")
            else:
                try:
                    gt = json.loads(entry["ground_truth"])
                    if "gt_parse" not in gt:
                        if self.verbose:
                            self.logger.warning(f"Отсутствует поле 'gt_parse' в ground_truth в файле {file_path} на строке {line_number}")
                        
                        if self.fix:
                            gt["gt_parse"] = {}
                            entry["ground_truth"] = json.dumps(gt)
                            return True, entry
                except json.JSONDecodeError:
                    if self.verbose:
                        self.logger.warning(f"Некорректный JSON в поле 'ground_truth' в файле {file_path} на строке {line_number}")
                    
                    if self.fix:
                        try:
                            if isinstance(entry["ground_truth"], str):
                                fixed_gt = entry["ground_truth"].replace("'", "\"")
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
                                        entry["ground_truth"] = json.dumps({"gt_parse": {}})
                                        return True, entry
                        except Exception:
                            if self.remove_broken:
                                self.logger.warning(f"Удаляем запись с невосстановимым JSON в поле 'ground_truth' в файле {file_path} на строке {line_number}")
                                return False, None
                            else:
                                entry["ground_truth"] = json.dumps({"gt_parse": {}})
                                return True, entry
            
            if missing_fields:
                if self.verbose:
                    self.logger.warning(f"Отсутствуют поля {', '.join(missing_fields)} в файле {file_path} на строке {line_number}")
                
                if self.fix:
                    if self.remove_broken and len(missing_fields) > 1:
                        self.logger.warning(f"Удаляем запись с отсутствующими обязательными полями в файле {file_path} на строке {line_number}")
                        return False, None
                    
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
            
            try:
                fixed_line = line
                
                if fixed_line.count('{') > fixed_line.count('}'):
                    fixed_line += '}' * (fixed_line.count('{') - fixed_line.count('}'))
                
                if fixed_line.count('"') % 2 != 0:
                    fixed_line += '"'
                
                try:
                    entry = json.loads(fixed_line)
                    self.logger.info(f"Успешно исправлен JSON в файле {file_path} на строке {line_number}")
                    
                    has_required_fields = True
                    if "file_name" not in entry:
                        has_required_fields = False
                    
                    if "ground_truth" not in entry:
                        has_required_fields = False
                    
                    if not has_required_fields and self.remove_broken:
                        self.logger.warning(f"Удаляем запись с отсутствующими обязательными полями в файле {file_path} на строке {line_number}")
                        return False, None
                    
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
                    backup_path = file_path + ".bak"
                    try:
                        import shutil
                        shutil.copy2(file_path, backup_path)
                        self.logger.info(f"Создан бэкап файла: {backup_path}")
                    except Exception as e:
                        self.logger.error(f"Не удалось создать бэкап файла: {e}")
                    
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
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            source_encoding = result['encoding']
            
            if source_encoding == target_encoding:
                print(f"Файл {file_path} уже в кодировке {target_encoding}")
                return True
            
            print(f"Конвертация файла {file_path} из {source_encoding} в {target_encoding}")
            
            content = raw_data.decode(source_encoding, errors='replace')
            
            backup_path = file_path + f".{source_encoding}.bak"
            import shutil
            shutil.copy2(file_path, backup_path)
            print(f"Создан бэкап файла: {backup_path}")
            
            with open(file_path, 'w', encoding=target_encoding) as f:
                f.write(content)
            
            print(f"Файл успешно конвертирован в {target_encoding}")
            return True
    
    except Exception as e:
        print(f"Ошибка при конвертации файла {file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Проверка и исправление JSONL файлов датасета Donut")
    parser.add_argument("dir", help="Путь к директории с JSONL файлами")
    parser.add_argument("--fix", action="store_true", help="Исправлять найденные ошибки")
    parser.add_argument("--verbose", action="store_true", help="Подробный вывод")
    parser.add_argument("--convert", action="store_true", help="Конвертировать файлы в UTF-8")
    parser.add_argument("--keep-broken", action="store_true", help="Сохранять невосстановимые записи, создавая для них новые")
    args = parser.parse_args()
    
    if args.convert:
        jsonl_files = []
        for root, _, files in os.walk(args.dir):
            for file in files:
                if file.endswith(".jsonl"):
                    jsonl_files.append(os.path.join(root, file))
        
        print(f"Найдено {len(jsonl_files)} JSONL файлов для конвертации")
        
        for file_path in jsonl_files:
            convert_file_encoding(file_path, 'utf-8')
    
    validator = JSONLValidator(fix=args.fix, verbose=args.verbose, remove_broken=not args.keep_broken)
    validator.validate_jsonl_files(args.dir)


if __name__ == "__main__":
    main()