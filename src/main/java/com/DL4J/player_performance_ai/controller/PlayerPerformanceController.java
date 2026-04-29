package com.DL4J.player_performance_ai.controller;

import com.DL4J.player_performance_ai.dto.PlayerPerformanceDto;
import com.DL4J.player_performance_ai.service.PlayerPerformanceService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@Slf4j
@RequiredArgsConstructor
@RestController
@RequestMapping("/api/performance")
public class PlayerPerformanceController {

    private final PlayerPerformanceService service;

    @GetMapping
    public List<PlayerPerformanceDto> getAll() {
        return service.getAll();
    }

    @PostMapping
    public PlayerPerformanceDto add(@RequestBody PlayerPerformanceDto dto) {
        return service.add(dto);
    }

    // Обновление существующих данных игрока по ID
    @PutMapping("/{id}")
    public ResponseEntity<PlayerPerformanceDto> update(
            @PathVariable Long id, @RequestBody PlayerPerformanceDto dto) {
        PlayerPerformanceDto updatedDto = service.update(id, dto);
        return updatedDto != null ?
                ResponseEntity.ok(updatedDto) :
                ResponseEntity.status(HttpStatus.NOT_FOUND).build();
    }

    // Удаление данных игрока по ID
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        boolean isDeleted = service.delete(id);
        return isDeleted ?
                ResponseEntity.noContent().build() :
                ResponseEntity.status(HttpStatus.NOT_FOUND).build();
    }


    /**
     * Эндпоинт для предсказания, подходит ли игрок на основе переданных метрик.
     * Пример входного JSON:
     * {
     *   "average": 50.5,
     *   "strikeRate": 140.0,
     *   "bowlingAverage": 20.0,
     *   "economyRate": 4.2,
     *   "fieldingStats": 15
     * }
     */
    @PostMapping("/predict")
    public boolean predictPlayer(@RequestBody Map<String, Float> playerMetrics) {
        float[] features = new float[]{
                playerMetrics.get("average"),
                playerMetrics.get("strikeRate"),
                playerMetrics.get("bowlingAverage"),
                playerMetrics.get("economyRate"),
                playerMetrics.get("fieldingStats")
        };
        return service.predictPlayerSuitability(features);
    }

    /**
     * Эндпоинт для запуска обучения модели с актуальными данными.
     * Переобучает AI-модель, используя все данные игроков из базы данных.
     */
    @PostMapping("/train")
    public ResponseEntity<String> trainModel() {
        try {
            service.trainModel();  // Вызов сервиса для обучения модели
            return ResponseEntity.ok("Model training started successfully.");
        } catch (Exception e) {
            log.error("Error during model training", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Failed to train the model.");
        }
    }
}