import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import axios from 'axios';
import * as employeService from '../../Service/EmployeService';


export const fetchDirections = createAsyncThunk('employee/fetchDirections',async () => {
        return await employeService.getDirections();
    }
);

const directionSlice = createSlice({
    name: 'direction',
    initialState: {
        directions: [],
    },
    reducers: {},
    extraReducers: (builder) => {
        builder
            .addCase(fetchDirections.fulfilled, (state, action) => {
                state.directions = action.payload;
            })
    },
});

export default directionSlice.reducer;
