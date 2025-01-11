import Foundation

struct Country: Codable {
    let code: String
    let name: String
}

class CountryLoader {
    
    static func loadCountries() -> [Country] {
        guard let url = Bundle.main.url(forResource: "countries", withExtension: "json") else {
            return []
        }
        do {
            let data = try Data(contentsOf: url)
            let countries  = try JSONDecoder().decode([Country].self, from: data)
            return countries
        } catch {
            print("Failed to load countries")
            return []
        }
    }
}
