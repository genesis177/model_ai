package com.DL4J.player_performance_ai.service;

import com.DL4J.player_performance_ai.ai.PlayerAIModel;
import com.DL4J.player_performance_ai.dto.PlayerPerformanceDto;
import com.DL4J.player_performance_ai.model.PlayerPerformance;
import com.DL4J.player_performance_ai.repository.PlayerPerformanceRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.*;
import org.springframework.stereotype.*;

import java.util.List;
import java.util.Optional;

@Service
@Slf4j
public class PlayerPerformanceService {

    private final PlayerPerformanceRepository repository;
    private final PlayerAIModel playerAIModel;
    private static final String MODEL_PATH = "src/main/resources/player_model.zip";

    @Autowired
    public PlayerPerformanceService(PlayerPerformanceRepository repository, PlayerAIModel playerAIModel) {
        this.repository = repository;
        this.playerAIModel = playerAIModel;
    }

    public List<PlayerPerformanceDto> getAll() {
        return repository.findAll().stream()
                .map(this::toDto)
                .toList();
    }

    public PlayerPerformanceDto add(PlayerPerformanceDto dto) {
        PlayerPerformance performance = repository.save(toEntity(dto));
        playerAIModel.trainModel();  // Обучение модели после добавления новых данных
        return toDto(performance);
    }

    /**
     * Обновление существующих данных игрока по ID.
     * @param id ID записи игрока для обновления.
     * @param dto Обновлённые данные игрока.
     * @return Обновлённый PlayerPerformanceDto или null, если запись не найдена.
     */
    public PlayerPerformanceDto update(Long id, PlayerPerformanceDto dto) {
        Optional<PlayerPerformance> existingPerformance = repository.findById(id);
        if (existingPerformance.isPresent()) {
            PlayerPerformance performance = existingPerformance.get();
            performance.setAverage(dto.getAverage());
            performance.setStrikeRate(dto.getStrikeRate());
            performance.setBowlingAverage(dto.getBowlingAverage());
            performance.setEconomyRate(dto.getEconomyRate());
            performance.setFieldingStats(dto.getFieldingStats());
            performance.setLabel(dto.getLabel());
            PlayerPerformance updatedPerformance = repository.save(performance);
            playerAIModel.trainModel();  // Обучение модели после обновления данных
            return toDto(updatedPerformance);
        }
        return null;
    }

    /**
     * Удаление данных игрока по ID.
     * @param id ID записи игрока для удаления.
     * @return true, если запись была удалена, иначе false.
     */
    public boolean delete(Long id) {
        if (repository.existsById(id)) {
            repository.deleteById(id);
            playerAIModel.trainModel();  // Обучение модели после удаления данных
            return true;
        }
        return false;
    }

    /**
     * Предсказывает, подходит ли игрок на основе переданных метрик.
     * @param features Массив метрик игрока.
     * @return true, если игрок подходит, иначе false.
     */
    public boolean predictPlayerSuitability(float[] features) {
        return playerAIModel.predict(features);
    }

    private PlayerPerformance toEntity(PlayerPerformanceDto dto) {
        PlayerPerformance entity = new PlayerPerformance();
        entity.setAverage(dto.getAverage());
        entity.setStrikeRate(dto.getStrikeRate());
        entity.setBowlingAverage(dto.getBowlingAverage());
        entity.setEconomyRate(dto.getEconomyRate());
        entity.setFieldingStats(dto.getFieldingStats());
        entity.setLabel(dto.getLabel());
        return entity;
    }

    private PlayerPerformanceDto toDto(PlayerPerformance entity) {
        PlayerPerformanceDto dto = new PlayerPerformanceDto();
        dto.setId(entity.getId());
        dto.setAverage(entity.getAverage());
        dto.setStrikeRate(entity.getStrikeRate());
        dto.setBowlingAverage(entity.getBowlingAverage());
        dto.setEconomyRate(entity.getEconomyRate());
        dto.setFieldingStats(entity.getFieldingStats());
        dto.setLabel(entity.getLabel());
        return dto;
    }

    public void trainModel() {
        playerAIModel.trainModel();
    }
}