import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import * as demandeService from '../../Service/DemandeService';

const initialState = {
  list: [],
  selectedDemande: null,
  status: 'idle',
  error: null,
};

export const fetchDemandes = createAsyncThunk('demandes/fetchDemandes', async () => {
  return await demandeService.getDemandes();
});

export const addDemande = createAsyncThunk('demandes/addDemande', async (formData) => {
  return await demandeService.addDemande(formData) ;
});

export const deleteDemande = createAsyncThunk('demandes/deleteDemande', async (id) => {
  return await demandeService.deleteDemande(id);
});

export const updateDemande = createAsyncThunk('demandes/updateDemande', async ({ id, demande }) => {
  return await demandeService.updateDemande(id, demande);
});


const demandesSlice = createSlice({
  name: 'demandes',
  initialState,
  reducers: {
    setSelectedDemande: (state, action) => {
      state.selectedDemande = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchDemandes.pending, (state) => {
        state.status = 'loading';
      })
      .addCase(fetchDemandes.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.list = action.payload;
      })
      .addCase(fetchDemandes.rejected, (state, action) => {
        state.status = 'failed';
        state.error = action.error.message;
      })
      .addCase(addDemande.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.list.push(action.payload);
      })
      .addCase(deleteDemande.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.list = state.list.filter((demande) => demande.demandeId !== action.payload);
      })
      .addCase(updateDemande.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.list = action.payload;
      })
  },
});

export default demandesSlice.reducer;
export const { setSelectedDemande } = demandesSlice.actions;