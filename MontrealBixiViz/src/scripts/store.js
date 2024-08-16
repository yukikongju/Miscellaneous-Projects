import { configureStore, createSlice } from "@reduxjs/toolkit";

const languageSlice = createSlice({
  name: "language",
  initialState: "en",
  reducers: {
    setLanguage: (state, action) => action.payload,
  },
});

const store = configureStore({
  reducer: {
    language: languageSlice.reducer,
  },
});

export default store;

// --- Language helpers

export const { setLanguage } = languageSlice.actions;

export function getCurrentLanguage() {
  return store.getState().language;
}

export function updateStoreLanguage(newLanguage) {
  store.dispatch(setLanguage(newLanguage));
}
