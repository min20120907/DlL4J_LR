package mrcnn1;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.NoSuchElementException;
import java.util.Scanner;

public class prac1 {
	public static void main(String args[]) throws FileNotFoundException {
		  int line = 0;
		  File file = new File("test.txt"); 
		  String cwd = System.getProperty("user.dir");
		  ArrayList<Integer> list = new ArrayList<Integer>();
		  Scanner myReader = new Scanner(file);
		  while(myReader.hasNext()) {
		  try {
			  list.add(Integer.parseInt(myReader.nextLine()));
		  }catch(NumberFormatException nfe) {
			  System.err.println("Error during parsing the file.");
			  nfe.printStackTrace();
			  break;
		  }
		  }
		  int sum=0;
		  System.out.println("Numbers: ");
		  for(int a =2;a<list.size();a++) {
			  sum+=list.indexOf(a);
			  System.out.println(list.indexOf(a));
			  
		  }
		  System.out.println("Average is: "+sum/(list.size()-2));
	}

	
	
}
