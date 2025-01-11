import Foundation

func fetchFlag(for countryCode: String, completion: @escaping (Result<Data, Error>) -> Void) {
    let urlString = "https://flagsapi.com/\(countryCode)/flat/64.png"
    guard let url = URL(string: urlString) else {
        completion(.failure(NSError(domain: "", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])))
        return
    }
    
    let task = URLSession.shared.dataTask(with: url) {
        data, response, error in if let error = error {
            completion(.failure(error))
            print("Error when fetching country: \(countryCode)")
        } else if let data = data {
            completion(.success(data))
        }
    }
        
    task.resume()
    
}
