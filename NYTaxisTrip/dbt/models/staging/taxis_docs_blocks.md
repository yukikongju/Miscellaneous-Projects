{% docs VendorID %}

A code indicating the LPEP provider that provided the record.

| VendorID | Definition                        |
| -------- | ----------                        |
| 1        | Creative Mobile Technologies, LLC |
| 2        | Curb Mobility, LLC                |
| 6        | Myle Technologies Inc             |
| 7        | Helix                             |

{% enddocs %}

{% docs RateCodeID %}

The final rate code in effect at the end of the trip.

| RateCodeID | Definition            |
| --------   | ----------            |
| 1          | Standard Rate         |
| 2          | JFK                   |
| 3          | Newark                |
| 4          | Nassau or Westchester |
| 5          | Negociated Fare       |
| 6          | Group Ride            |
| 99         | Null/unknown          |

{% enddocs %}

{% docs PaymentType %}

A numeric code signifying how the passenger paid for the trip.

| Payment Type Code | Description     |
|-------------------|-----------------|
| 0                 | Flex Fare trip  |
| 1                 | Credit card     |
| 2                 | Cash            |
| 3                 | No charge       |
| 4                 | Dispute         |
| 5                 | Unknown         |
| 6                 | Voided trip     |

{% enddocs %}


{% docs TripType %}

A code indicating whether the trip was a street-hail or a dispatch that is
automatically assigned based on the metered rate in use but can be altered
by the driver

| Trip Type Code | Description |
|----------------|-------------|
| 1              | Street-hail |
| 2              | Dispatch    |

{% enddocs %}
