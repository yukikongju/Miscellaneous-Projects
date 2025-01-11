//
//  ContentView.swift
//  WeSplit
//
//  Created by Emulie Chhor on 2025-01-10.
//

import SwiftUI

struct ContentView: View {
    
    @State private var tapCount = 0
    @State private var name = ""
    @State private var selectedStudent = "ENC"
    let students = ["ENC", "LEX", "HDS", "TJP"]
    
    @State private var numPeople: Int = 1
    @State private var billAmount: Double = 0
    @State private var tipPercentage = 15.0
    let tipPercentages: [Double] = [0.0, 5.0, 10.0, 15.0, 20.0]
    
    @State private var username = ""
    
    var body: some View {
        
        VStack {
            Image(systemName: "globe")
                .imageScale(.small)
                .foregroundStyle(.tint)
            Text("Hello, world!")
        }
        .padding()
        
        Form {
            Section {
                Text("Hello buddy!!")
                Text("Hello buddy!!")
                Text("Hello buddy!!")
            }
            Section {
                Button("Tap Count: \(tapCount)") {
                    tapCount += 1

                }
            }
            
            Section {
                Text("Write your name: \(name)")
                TextField("Name", text: $name)
            }
            
            Section {
                ForEach(0 ..< 10) { number in
                    Text("Row \(number)")
                }
            }
            
            Section {
                Picker("Select a user", selection: $selectedStudent) {
                    ForEach(students, id: \.self) {
                        Text($0)
                    }
                }

            }
            
            Section {
                // Calculate check amount for each person
                Text("Bill Calculator")
                Picker("Number of people", selection: $numPeople) {
                    ForEach(1 ..< 10) { number in
                        Text("\(number)")
                    }
                }
                
                Picker("Tip Percentage", selection: $tipPercentage) {
                    ForEach(tipPercentages, id: \.self) { perc in
                        Text(String(format: "%.2f%%", perc))
                    }
                }
                .pickerStyle(.segmented)
                
//                Text("Bill Amount: \(billAmount)", format: .currency(code: Locale.current.currency?.identifier ?? "USD"))

                
                TextField("Bill Amount", value: $billAmount, format: .currency(code: Locale.current.currency?.identifier ?? "USD"))
                    .keyboardType(.decimalPad)
                

                
                // Tip Calculation
                let tipAmount = billAmount * Double(tipPercentage) / 100.0
                Text(String(format: "Tip Amount: $%.2f%", tipAmount))
                
                // Amount per person
                let amountPerPerson = billAmount / Double(numPeople)
                Text(String(format: "Amount per person: $%.2f", amountPerPerson))
                
                // Amount per person w/ tip included
                let amountPerPersonWithTip = (billAmount + tipAmount) / Double(numPeople)
                Text(String(format: "Amount per person with Tip: $%.2f%", amountPerPersonWithTip))
                
            }
        
        }
        
        
//    NavigationStack {
//        Form {
//            Section {
//                Text("Hello buddy!!")
//            }
//        }
//    }
//    .navigationTitle("SwiftUI")
//    .navigationBarTitleDisplayMode(.inline)
        
             
    }
        

        


    
}

#Preview {
    ContentView()
}
