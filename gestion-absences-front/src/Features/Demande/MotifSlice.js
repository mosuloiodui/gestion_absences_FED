import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import * as demandeService from '../../Service/DemandeService';


export const fetchMotifs = createAsyncThunk('demandes/fetchMotifs',async () => {
        return await demandeService.getMotifs();
    }
);

const motifSlice = createSlice({
    name: 'motif',
    initialState: {
        motifs: [],
    },
    reducers: {},
    extraReducers: (builder) => {
        builder
            .addCase(fetchMotifs.fulfilled, (state, action) => {
                state.motifs = action.payload;
            })
    },
});

export default motifSlice.reducer;
