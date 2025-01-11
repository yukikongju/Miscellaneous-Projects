//
//  ContentView.swift
//  GuessTheFlag
//
//  Created by Emulie Chhor on 2025-01-10.
//

import SwiftUI

struct ContentView: View {
    
    @State private var score: Int = 0


    var numChoices: Int = 5
    var countries: [Country] = CountryLoader.loadCountries()

    // TODO: get number of countries
    //    let numCountries: Int = countries.count
    let numCountries: Int = 214

    
    init() {
        self.countries = CountryLoader.loadCountries()
//        self.selectedCountry = selectRandomCountry()
//        self.otherCountries = selectOtherCountries()
        self.numChoices = 5
        
    }
    // TODO: initialize first country randomly
    //    @State private var selectedCountry: Country = selectRandomCountry()
    @State private var selectedCountry: Country = Country(code: "US", name: "United States")
    @State private var otherCountries: [Country]
    
    
    func selectRandomCountry() -> Country {
        let randomIndex = Int.random(in: 0..<numCountries)
        return countries[randomIndex]
    }
    
    
    func selectOtherCountries() -> [Country] {
        var listCountries: [Country] = []
        
        while listCountries.count < numChoices {
            let country = selectRandomCountry()
            if country.name != selectedCountry.name {
                listCountries.append(country)
            }
        }
        
        return listCountries
    }
    
//    @State private var otherCountries = selectOtherCountries()
    
    
    var body: some View {
        VStack {
            Text("**Guess the Flag**")
        }
        .padding()
        
        Form {
            Section {
            // TODO: Show Image centered
                AsyncImage(url: URL(string: "https://flagsapi.com/\(selectedCountry.code)/flat/64.png"))
            }
            
            Section {
            // TODO: Stack Button with choice of Country
            
            }
        }
        
        
        VStack {
            Text("Score: \(score)")
        }
        
        
    }
}

#Preview {
    ContentView()
}
