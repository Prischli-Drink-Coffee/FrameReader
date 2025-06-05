-- phpMyAdmin SQL Dump
-- version 5.2.0
-- https://www.phpmyadmin.net/
--
-- Хост: 127.0.0.1:3306
-- Время создания: Июн 05 2025 г., 02:24
-- Версия сервера: 5.7.39
-- Версия PHP: 7.2.34

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- База данных: `FrameReader`
--

-- --------------------------------------------------------

--
-- Структура таблицы `frame_annotations`
--

CREATE TABLE `frame_annotations` (
  `id` int(11) NOT NULL,
  `video_session_id` int(11) NOT NULL,
  `frame_timestamp` decimal(10,3) NOT NULL,
  `annotation_data` json NOT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Структура таблицы `users`
--

CREATE TABLE `users` (
  `id` int(11) NOT NULL,
  `fingerprint_hash` varchar(64) NOT NULL,
  `first_visit` timestamp NOT NULL,
  `last_activity` timestamp NOT NULL,
  `total_sessions` int(11) NOT NULL DEFAULT '0',
  `total_videos_processed` int(11) NOT NULL DEFAULT '0',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Структура таблицы `user_sessions`
--

CREATE TABLE `user_sessions` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `jwt_token_hash` varchar(64) NOT NULL,
  `expires_at` timestamp NOT NULL,
  `user_agent` text,
  `ip_address` varchar(45) DEFAULT NULL,
  `is_active` tinyint(1) NOT NULL DEFAULT '1',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Структура таблицы `video_sessions`
--

CREATE TABLE `video_sessions` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `video_url` text NOT NULL,
  `processing_status` enum('processing','completed','failed') NOT NULL DEFAULT 'processing',
  `started_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `completed_at` timestamp NULL DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Индексы сохранённых таблиц
--

--
-- Индексы таблицы `frame_annotations`
--
ALTER TABLE `frame_annotations`
  ADD PRIMARY KEY (`id`),
  ADD KEY `idx_frame_annotations_video` (`video_session_id`,`frame_timestamp`);

--
-- Индексы таблицы `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `fingerprint_hash` (`fingerprint_hash`),
  ADD KEY `idx_users_fingerprint` (`fingerprint_hash`);

--
-- Индексы таблицы `user_sessions`
--
ALTER TABLE `user_sessions`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `jwt_token_hash` (`jwt_token_hash`),
  ADD KEY `idx_user_sessions_active` (`user_id`,`is_active`,`expires_at`);

--
-- Индексы таблицы `video_sessions`
--
ALTER TABLE `video_sessions`
  ADD PRIMARY KEY (`id`),
  ADD KEY `idx_video_sessions_status` (`user_id`,`processing_status`);

--
-- AUTO_INCREMENT для сохранённых таблиц
--

--
-- AUTO_INCREMENT для таблицы `frame_annotations`
--
ALTER TABLE `frame_annotations`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT для таблицы `users`
--
ALTER TABLE `users`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT для таблицы `user_sessions`
--
ALTER TABLE `user_sessions`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT для таблицы `video_sessions`
--
ALTER TABLE `video_sessions`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- Ограничения внешнего ключа сохраненных таблиц
--

--
-- Ограничения внешнего ключа таблицы `frame_annotations`
--
ALTER TABLE `frame_annotations`
  ADD CONSTRAINT `frame_annotations_ibfk_1` FOREIGN KEY (`video_session_id`) REFERENCES `video_sessions` (`id`) ON DELETE CASCADE;

--
-- Ограничения внешнего ключа таблицы `user_sessions`
--
ALTER TABLE `user_sessions`
  ADD CONSTRAINT `user_sessions_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE;

--
-- Ограничения внешнего ключа таблицы `video_sessions`
--
ALTER TABLE `video_sessions`
  ADD CONSTRAINT `video_sessions_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
