package com.DL4J.player_performance_ai.ai;

import com.DL4J.player_performance_ai.model.PlayerPerformance;
import com.DL4J.player_performance_ai.repository.PlayerPerformanceRepository;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@Component
@Slf4j
public class PlayerAIModel {

    private final PlayerPerformanceRepository playerPerformanceRepository;
    private final MultiLayerNetwork model;
    private static final String MODEL_PATH = "src/main/resources/player_model.zip";

    public PlayerAIModel(PlayerPerformanceRepository repository) {
        this.playerPerformanceRepository = repository;
        this.model = initializeModel();  // Инициализация или загрузка модели
    }

    private MultiLayerNetwork initializeModel() {
        File modelFile = new File(MODEL_PATH);
        if (modelFile.exists()) {
            try {
                System.out.println("Loaded existing model from " + MODEL_PATH);
                return ModelSerializer.restoreMultiLayerNetwork(modelFile);
            } catch (IOException e) {
                System.err.println("Failed to load the model. Creating a new one.");
            }
        }
        return createNewModel();
    }

    private MultiLayerNetwork createNewModel() {
        MultiLayerNetwork newModel = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(42)
                .updater(new Adam(0.001))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(5).nOut(64).activation(Activation.RELU).build())
                .layer(1, new DenseLayer.Builder().nIn(64).nOut(32).activation(Activation.RELU).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SIGMOID).nIn(32).nOut(1).build())
                .build());
        newModel.init();
        saveModel(newModel);
        return newModel;
    }

    private void saveModel(MultiLayerNetwork model) {
        try {
            ModelSerializer.writeModel(model, MODEL_PATH, true);
            System.out.println("Model saved to " + MODEL_PATH);
        } catch (IOException e) {
            System.err.println("Failed to save the model.");
        }
    }

    public void trainModel() {
        List<PlayerPerformance> players = playerPerformanceRepository.findAll();  // Получение всех данных игроков
        List<DataSet> dataSets = new ArrayList<>();

        for (PlayerPerformance player : players) {
            // Преобразование данных игрока в 2D массив (1 строка, 5 столбцов)
            INDArray features = Nd4j.create(new float[][]{
                    {
                            (float) player.getAverage(),
                            (float) player.getStrikeRate(),
                            (float) player.getBowlingAverage(),
                            (float) player.getEconomyRate(),
                            player.getFieldingStats()
                    }
            });

            // Преобразование метки (label) в 2D массив (1 строка, 1 столбец)
            INDArray label = Nd4j.create(new float[][]{
                    {player.getLabel()}
            });

            // Создание DataSet с корректной формой признаков и метки
            dataSets.add(new DataSet(features, label));
        }

        // Создание ListDataSetIterator из списка DataSet
        ListDataSetIterator<DataSet> iterator = new ListDataSetIterator<>(dataSets);

        // Обучение модели в течение 50 эпох
        model.fit(iterator, 50);

        // Сохранение переобученной модели
        saveModel(model);
    }


    public boolean predict(float[] features) {
        // Преобразование входного массива в 2D матрицу (1 строка, 5 столбцов)
        INDArray input = Nd4j.create(features).reshape(1, features.length);

        // Получение результата от модели
        INDArray output = model.output(input);

        // Предполагается, что модель возвращает вероятность или класс
        return output.getFloat(0) > 0.5;
    }
}