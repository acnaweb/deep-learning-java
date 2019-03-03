package br.com.marketmining.dsaflow;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.univocity.parsers.csv.CsvParser;
import com.univocity.parsers.csv.CsvParserSettings;

public class DatasetUtil {
	public static INDArray getHousingInstances() {
		Path path = Paths.get("data", "housing.data.txt");
		CsvParserSettings settings = new CsvParserSettings();

		CsvParser parser = new CsvParser(settings);
		List<String[]> allRows = parser.parseAll(path.toFile());

		String[][] w = allRows.toArray(new String[][] {});

		double[][] array = new double[w.length][w[0].length];
		
		return Nd4j.create(array);

	}
	
	public static INDArray getAndInstances() {
		Path path = Paths.get("data", "and.data.txt");
		CsvParserSettings settings = new CsvParserSettings();

		CsvParser parser = new CsvParser(settings);
		List<String[]> allRows = parser.parseAll(path.toFile());

		String[][] w = allRows.toArray(new String[][] {});

		double[][] array = new double[w.length][w[0].length];
		
		return Nd4j.create(array);

	}
}
