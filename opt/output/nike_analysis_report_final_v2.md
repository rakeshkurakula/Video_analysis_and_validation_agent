# Deviation Report

- Scenario: Search_for_black_running_shoes,_verify_size_10_availability,_and_add_to_cart
- Run ID: run_20260104_152003
- Steps analyzed: 15
- Deviations: 11
- Average confidence: 80.7%
- Status counts: Hallucinated 6, Deviation-Skipped 5, Observed 4
- Proofs video: /Users/rakeshk94/Desktop/test_zeus/Video_analysis_and_validation_agent/opt/proofs/Search_for_black_running_shoes,_verify_size_10_availability,_and_add_to_cart/run_20260104_152003/videos/video_of_Search_for_black_running_shoes,_verify_size_10_availability,_and_add_to_cart.webm

## Final Output
- Failure message: EXPECTED RESULT: Visual signals ['Results', 'products', 'Found'] present on search results page.
ACTUAL RESULT: Detected signals ['Results', 'products']; 'Found' missing.

## Step Results
| Step | Description | Result | Conf | Notes | Evidence |
| --- | --- | --- | --- | --- | --- |
| 1 | the user navigates to "https://www.nike.com/in/" | Hallucinated | 70% | Log claims step was executed but no visual evidence found. | - |
| 2 | the user waits for the homepage to fully load | Hallucinated | 70% | Log claims step was executed but no visual evidence found. | - |
| 3 | the user closes any promotional popup or banner if... | Deviation-Skipped | 90% | Previous step failed; no evidence for this step. | - |
| 4 | the user clicks the search icon or focuses the sea... | Observed | 90% | Visual evidence found in 9 frame(s). | /Users/rakeshk94/Desktop/test_zeus/Video_analysis_and_validation_agent/opt/proofs/Search_for_black_running_shoes,_verify_size_10_availability,_and_add_to_cart/run_20260104_152003/screenshots/press_key_combination_end_1767520270875638000.png, /Users/rakeshk94/Desktop/test_zeus/Video_analysis_and_validation_agent/opt/proofs/Search_for_black_running_shoes,_verify_size_10_availability,_and_add_to_cart/run_20260104_152003/screenshots/entertext_start_1767520269322885000.png |
| 5 | the user types "black running shoes" into the sear... | Observed | 90% | Visual evidence found in 13 frame(s). | /Users/rakeshk94/Desktop/test_zeus/Video_analysis_and_validation_agent/opt/proofs/Search_for_black_running_shoes,_verify_size_10_availability,_and_add_to_cart/run_20260104_152003/screenshots/press_key_combination_end_1767520272694317000.png, /Users/rakeshk94/Desktop/test_zeus/Video_analysis_and_validation_agent/opt/temp_frames/run_20260104_152003/frame_0033.png |
| 6 | the user submits the search query | Hallucinated | 70% | Log claims step was executed but no visual evidence found. | - |
| 7 | search results page should show matching products | Hallucinated | 70% | Log claims step was executed but no visual evidence found. | - |
| 8 | the user clicks on the first matching black runnin... | Hallucinated | 70% | Log claims step was executed but no visual evidence found. | - |
| 9 | the product detail page should display size option... | Deviation-Skipped | 90% | Previous step failed; no evidence for this step. | - |
| 10 | the user checks if "size 10" is available | Deviation-Skipped | 90% | Previous step failed; no evidence for this step. | - |
| 11 | the user selects size "10" | Observed | 80% | Visual evidence found in 13 frame(s). | /Users/rakeshk94/Desktop/test_zeus/Video_analysis_and_validation_agent/opt/proofs/Search_for_black_running_shoes,_verify_size_10_availability,_and_add_to_cart/run_20260104_152003/screenshots/openurl_end_1767520219568692000.png, /Users/rakeshk94/Desktop/test_zeus/Video_analysis_and_validation_agent/opt/proofs/Search_for_black_running_shoes,_verify_size_10_availability,_and_add_to_cart/run_20260104_152003/screenshots/click_end_1767520261559565000.png |
| 12 | the user clicks the "Add to Bag" button | Hallucinated | 70% | Log claims step was executed but no visual evidence found. | - |
| 13 | the product should be added to the bag | Deviation-Skipped | 90% | Previous step failed; no evidence for this step. | - |
| 14 | the user navigates to the cart page | Deviation-Skipped | 90% | Previous step failed; no evidence for this step. | - |
| 15 | the cart should display the product with size "10"... | Observed | 80% | Visual evidence found in 13 frame(s). | /Users/rakeshk94/Desktop/test_zeus/Video_analysis_and_validation_agent/opt/proofs/Search_for_black_running_shoes,_verify_size_10_availability,_and_add_to_cart/run_20260104_152003/screenshots/openurl_end_1767520219568692000.png, /Users/rakeshk94/Desktop/test_zeus/Video_analysis_and_validation_agent/opt/proofs/Search_for_black_running_shoes,_verify_size_10_availability,_and_add_to_cart/run_20260104_152003/screenshots/click_end_1767520261559565000.png |
